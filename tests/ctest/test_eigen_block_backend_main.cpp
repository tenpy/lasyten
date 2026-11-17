// Entry point of the Python-free Eigen block-backend test (LASYTEN_WITH_PYTHON=OFF builds).
// With Python, the same test is run through CommonCppTests (create_test_sourcelist).

int test_eigen_block_backend(int argc, char** argv);

int
main(int argc, char** argv)
{
    return test_eigen_block_backend(argc, argv);
}
