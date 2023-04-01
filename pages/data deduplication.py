from collections import defaultdict, deque

import desbordante as desb
import pandas as pd
import streamlit as st

# st.set_page_config(layout="wide")

"""
# Deduplication scenario
***
"""


with st.sidebar:
    ALGORITHM = st.selectbox("Algorithm", ["Pyro"])
    ERROR = st.slider("Error", 0.00001, 0.01, 0.001, step=0.000001, format="%f")
    CONFIG = {"error": ERROR, "max_lhs": 1}
    DATASET_PATH = "duplicates.csv"
    HEADER = 0
    SEPARATOR = ","
    OUTPUT_FILE = "output.csv"
    WINDOW_SIZE = st.number_input("Window size", value=4)

    CONFIG_STRING = f"""
    {ALGORITHM=}
    {ERROR=:.8f}
    {DATASET_PATH=}
    {SEPARATOR=}
    {WINDOW_SIZE=}
    """

form_number = 0


@st.cache_data(show_spinner=True)
def get_1lhs_fds(df, algo_name, algo_config):
    algo = getattr(desb, algo_name)()
    algo.fit(df, **algo_config)
    algo.execute(**algo_config)
    return sorted(
        (lhs_indices[0], fd.rhs_index)
        for fd in algo.get_results()
        if len(lhs_indices := fd.lhs_indices) == 1
    )


@st.cache_data(show_spinner=True)
def get_lhs_from_sorted_fds(fds):
    lhs = []
    prev_lhs = None
    for cur_lhs, _ in fds:
        if cur_lhs != prev_lhs:
            lhs.append(cur_lhs)
        prev_lhs = cur_lhs
    return lhs


@st.cache_data(show_spinner=True)
def count_matches(row1, row2, rhs: list[int]):
    return sum(row1[index] == row2[index] for index in rhs)


def print_fd_info(df: pd.DataFrame, fds: list[tuple[int, int]]):
    fd_dict = defaultdict(list)
    for lhs, rhs in fds:
        fd_dict[lhs].append(df.columns[rhs])
    "#### AFD info:"
    st.text(
        "\n".join(
            f'{lhs}: {df.columns[lhs]} -> ( {" ".join(fd_dict[lhs])} )'
            for lhs in get_lhs_from_sorted_fds(fds)
        )
    )


@st.cache_data(show_spinner=True)
def keepall_handler(df, new_rows, remaining_rows, used_rows):
    new_rows.extend(df.iloc[list(remaining_rows)].itertuples(index=False))
    remaining_rows.clear()
    return remaining_rows


def drop_handler(df, new_rows, remaining_rows, used_rows):
    indices_to_add = list(remaining_rows - used_rows)
    new_rows.extend(df.iloc[indices_to_add].itertuples(index=False))
    remaining_rows.clear()
    return remaining_rows


def choose_index(col_name, distinct_values):
    global form_number
    st.write(f"Column: {col_name}.")
    index = st.radio(
        "Which value to use?",
        label_visibility="collapsed",
        key="radio" + str(form_number),
        options=enumerate(distinct_values),
        horizontal=True,
        format_func=lambda x: x[1],
    )[0]
    form_number += 1

    return index


def merge_handler(df, new_rows, remaining_rows, used_rows):
    if not used_rows:
        return
    new_row = []
    st.markdown("## Which values to use?")
    for col_name, values in zip(
        df.columns, zip(*df.iloc[list(used_rows)].itertuples(index=False))
    ):
        distinct_values = sorted(list(set(values)), key=lambda x: str(x))
        index = (
            0 if len(distinct_values) == 1 else choose_index(col_name, distinct_values)
        )
        new_row.append(distinct_values[index])
    remaining_rows -= used_rows
    new_rows.append(new_row)
    return remaining_rows


def unknown_handler(df, new_rows, remaining_rows, used_rows):
    st.error("Unknown command.")
    st.stop()


def ask_rows(df: pd.DataFrame, window: deque[tuple[int, object]]) -> list:
    global form_number

    commands = {
        "keepall": keepall_handler,
        "drop": drop_handler,
        "merge": merge_handler,
    }
    remaining_rows = {row_info[0] for row_info in window}
    new_rows = []

    while remaining_rows:
        st.dataframe(df.iloc[sorted(remaining_rows)])
        inp = ""
        with st.form("Submit command " + str(form_number)):
            inp = st.text_input("Command: ", placeholder="keepall, drop or merge")
            st.form_submit_button("Run")

        if not inp:
            st.stop()

        form_number += 1

        command_args = inp.split()
        if not command_args:
            st.text("Please input a command!")
            continue

        command, *used_rows = command_args
        used_rows = {
            col_num for col in used_rows if (col_num := int(col)) in remaining_rows
        }
        remaining_rows = commands.get(command, unknown_handler)(
            df, new_rows, remaining_rows, used_rows
        )

    return new_rows


@st.cache_data(show_spinner=True)
def is_similar(row_info, window, chosen_cols, matches_required):
    return any(
        count_matches(prev_row_info[1], row_info[1], chosen_cols) >= matches_required
        for prev_row_info in window
    )


# @st.cache_data(show_spinner=True, experimental_allow_widgets=True)
def get_deduped_rows(
    df: pd.DataFrame,
    chosen_cols: list[int],
    matches_required: int,
    fds: list[tuple[int, int]],
):
    df.sort_values(
        [df.columns[rhs_col] for _, rhs_col in fds if rhs_col in chosen_cols],
        inplace=True,
    )
    df.reset_index(inplace=True, drop=True)

    window = deque()
    new_rows = []
    has_duplicate = False
    for row_info in df.iterrows():
        if len(window) < WINDOW_SIZE:
            if not has_duplicate:
                has_duplicate = is_similar(
                    row_info, window, chosen_cols, matches_required
                )
        elif not has_duplicate:
            new_rows.append(window.pop()[1].values)
            has_duplicate = is_similar(row_info, window, chosen_cols, matches_required)
        elif not is_similar(row_info, window, chosen_cols, matches_required):
            new_rows.extend(ask_rows(df, window))
            window.clear()
            has_duplicate = False
        window.appendleft(row_info)

    new_rows.extend(
        ask_rows(df, window)
        if has_duplicate
        else (row_info[1].values for row_info in window)
    )
    return new_rows


@st.cache_data(show_spinner=True)
def open_df(path):
    return pd.read_csv(path, sep=SEPARATOR, header=HEADER, dtype=str, index_col=False)


def main():
    "## Deduplication parameters:"
    st.text(CONFIG_STRING)

    df = open_df(DATASET_PATH)
    "## Dataset sample:"
    st.dataframe(df.head(10))
    f"#### Original records: {len(df)}"

    fds = get_1lhs_fds(df, ALGORITHM, CONFIG)
    print_fd_info(df, fds)

    "#### LHS column to use"
    lhs_column_name = st.selectbox(
        label="",
        label_visibility="collapsed",
        options=set(map(lambda x: df.columns[x[0]], fds)),
    )
    lhs_column = df.columns.get_loc(lhs_column_name)

    fds = list(filter(lambda fd: fd[0] == lhs_column, fds))
    if not fds:
        st.write("No FDs with this LHS!")
        return
    else:
        pass
        rhs_list = [df.columns[rhs] for _, rhs in fds]
        "#### RHS columns to use"
        selected_rhs = st.multiselect(
            "", label_visibility="collapsed", options=rhs_list
        )
        chosen_cols = sorted(list(map(lambda x: df.columns.get_loc(x), selected_rhs)))
        "#### Equal columns to consider duplicates:"
        matches_required = st.number_input("", label_visibility="collapsed", value=2)

        with st.spinner("Please, wait"):
            new_rows = get_deduped_rows(df, chosen_cols, matches_required, fds)

        st.text(
            f"Resulting records: {len(new_rows)}. Duplicates found: {len(df) - len(new_rows)}"
        )
        new_df = pd.DataFrame(new_rows, columns=df.columns)

        st.markdown("## Final dataframe")
        st.dataframe(new_df)


if __name__ == "__main__":
    main()
