import desbordante as desb
import pandas as pd
import streamlit as st


"""
# Anomaly detection scenario
***
"""

with st.sidebar:
    ERROR = [0.01, 0.03, 0.05]
    HEADER = 0
    SEPARATOR = ","

    EXACT_ALGORITHM = st.selectbox("Exact algorithm", ["FastFDs"])
    EXACT_CONFIG = {}
    APPROXIMATE_ALGORITHM = st.selectbox("Approximate algorithm", ["Pyro"])
    APPROXIMATE_CONFIG = {"error": ERROR}
    DATASET_PATH = "./cargo/"

    METRIC = st.selectbox("Metric", ["euclidean", "levenshtein", "cosine"])
    METRIC_algorithm = st.selectbox("Metric algorithm", ["brute", "approx", "calipers"])

    DISTANCE = st.slider("Distance", 0, 10, 4)

    METRIC_VERIFIER = "MetricVerifier"
    METRIC_VERIFIER_CONFIG = {
        "lhs_indices": [1],
        "rhs_indices": [3],
        "metric": METRIC,
        "metric_algorithm": METRIC_algorithm,
        "parameter": DISTANCE,
    }

# available metrics: euclidean, levenshtein, cosine
# available metric algorithms: brute, approx, calipers
# parameter: desired distance related to metric. e.g., for euclidean metric parameter=4 means that euclidean distance has to be no larger than 4


CONFIG_STRING = f"""
{ERROR=}
{DATASET_PATH=}
{HEADER=}
{SEPARATOR=}
{EXACT_ALGORITHM=}
{APPROXIMATE_ALGORITHM=}
{METRIC_VERIFIER=}"""


def get_result_set_fd(df, algo_name, algo_config):
    algo = getattr(desb, algo_name)()
    algo.fit(df, **algo_config)
    algo.execute(**algo_config)

    return {(tuple(fd.lhs_indices), fd.rhs_index) for fd in algo.get_results()}


def get_result_set_mv(df, algo_name, algo_config):
    algo = getattr(desb, algo_name)()
    algo.fit(df, **algo_config)
    algo.execute(**algo_config)

    return algo.get_results()


def res_to_col_names(res, int_to_col_name):
    r = []

    for fd in res:
        r.append(
            (
                [int_to_col_name[lhs_index] for lhs_index in fd[0]],
                int_to_col_name[fd[1]],
            )
        )

    r.sort()

    return r


def print_fd_set(fd_set):
    outp_string = "\n".join([", ".join(fd[0]) + " -> " + str(fd[1]) for fd in fd_set])
    st.text(outp_string)


def diff(fd_set_1, fd_set_2, int_to_col_name):
    diff = fd_set_1 - fd_set_2

    if diff:
        st.warning("Found missing FDs")
        "#### Missing FDs:"
        print_fd_set(res_to_col_names(diff, int_to_col_name))
    else:
        st.info("No missing FDs.")

    return diff


def main():
    st.text(CONFIG_STRING)
    df1 = pd.read_csv(DATASET_PATH + "cargo_data_1.csv", sep=SEPARATOR, header=HEADER)
    df2 = pd.read_csv(DATASET_PATH + "cargo_data_2.csv", sep=SEPARATOR, header=HEADER)
    df3 = pd.read_csv(DATASET_PATH + "cargo_data_3.csv", sep=SEPARATOR, header=HEADER)

    tab1, tab2, tab3 = st.tabs(["Dataset 1", "Dataset 2", "Dataset 3"])

    with tab1:
        st.markdown("#### Dataset №1")
        st.dataframe(df1)
    with tab2:
        st.markdown("#### Dataset №2")
        st.dataframe(df2)
    with tab3:
        st.markdown("#### Dataset №3")
        st.dataframe(df3)

    int_to_col_name = {i: df1.columns[i] for i in range(len(df1.columns))}

    "## Mine FDs for D1"

    fd_res_1 = get_result_set_fd(df1, EXACT_ALGORITHM, EXACT_CONFIG)
    r1 = res_to_col_names(fd_res_1, int_to_col_name)
    print_fd_set(r1)

    "## Mine FDs for D2"

    fd_res_2 = get_result_set_fd(df2, EXACT_ALGORITHM, EXACT_CONFIG)
    r2 = res_to_col_names(fd_res_2, int_to_col_name)
    print_fd_set(r2)

    "## Check whether some of FDs are missing"
    diff12 = diff(fd_res_1, fd_res_2, int_to_col_name)

    if not diff12:
        "#### Diff is empty, proceed to D3"

        "## Mine FDs for D3"

        fd_res_3 = get_result_set_fd(df3, EXACT_ALGORITHM, EXACT_CONFIG)
        r3 = res_to_col_names(fd_res_3, int_to_col_name)
        print_fd_set(r3)

        # "## Missing FD found here"
        diff23 = diff(fd_res_2, fd_res_3, int_to_col_name)

        if diff23:
            "## Initiate processes for checking if missing FD has become an AFD"

            for error in ERROR:
                st.write("Error: ", error)
                APPROXIMATE_CONFIG["error"] = error
                afd_res = get_result_set_fd(
                    df3, APPROXIMATE_ALGORITHM, APPROXIMATE_CONFIG
                )
                print_fd_set(res_to_col_names(afd_res, int_to_col_name))

                if diff in afd_res:
                    st.markdown("**Missing FD is an AFD.**")
                else:
                    st.markdown("**Missing FD is not an AFD.**")

                st.markdown("***")

            "## Missing FD is not a part of AFD set. proceed to MFD validation phase"

            "### Check the stats of RHS attribute"
            with st.echo():
                st.dataframe(df3["item_weight"].describe())
            
            # upper_bound = st.selectbox("Select upper bound", list(df3["item_weight"].describe().to_dict().keys()))

            "### Define range for MetricVerifier parameter as [1; std]"

            with st.echo():
                for pj in range(1, int(df3["item_weight"].std())):
                    METRIC_VERIFIER_CONFIG["parameter"] = pj
                    mv_res = get_result_set_mv(df3, METRIC_VERIFIER, METRIC_VERIFIER_CONFIG)

                    if mv_res:
                        st.write("**MFD with parameter {} holds.**".format(pj))
                        break
                    else:
                        st.write("**MFD with parameter {} not holds.**".format(pj))


if __name__ == "__main__":
    main()
