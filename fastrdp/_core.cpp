#include "Eigen/Dense"
#include <format>
#include <string>
#include <vector>
#include <variant>
#include <cmath>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/ndarrayobject.h>


namespace E = Eigen;
using std::string;
using std::vector;
using DStride = E::Stride<E::Dynamic, E::Dynamic>;
using RMatrixXd = E::Matrix<double, E::Dynamic, E::Dynamic, E::RowMajor>;
using MapMatrix = E::Map<RMatrixXd, 0, DStride>;


void destruct_capsule(PyObject* cap) {
    void *ptr = PyCapsule_GetPointer(cap, NULL);
    delete[] static_cast<double*>(ptr);
}


struct MatOrStr {
    std::variant<MapMatrix, string> data;
    bool is_mat() {
        return data.index() == 0;
    }
};


MapMatrix new_mat(E::Index rows, E::Index cols) {
    double *data = new double[rows * cols];
    MapMatrix mat(data, rows, cols, DStride(cols, 1));
    return mat;
}


void del_mat(MapMatrix& m) {
    delete[] m.data();
}


MatOrStr to_matrix(PyArrayObject* x) {
    int x_ndim = PyArray_NDIM(x);
    if (x_ndim != 2) {
        const string err = std::format("The array should have 2 dims, but has {} dims.", x_ndim);
        return {err};
    }
    char& x_type = PyArray_DESCR(x)->type;
    if (x_type != 'd') {
        const string err = std::format("The array should have type d, but has type {}.", x_type);
        return {err};
    }

    npy_intp r = PyArray_DIM(x, 0), c = PyArray_DIM(x, 1);
    npy_intp sr = PyArray_STRIDE(x, 0), sc = PyArray_STRIDE(x, 1);
    double *x_data = static_cast<double *>(PyArray_DATA(x));
    const DStride x_mat_s(sr / 8, sc / 8);
    MapMatrix x_mat(x_data, r, c, x_mat_s);
    return {x_mat};
}


PyArrayObject* to_ndarray(MapMatrix& m) {
    npy_intp const dims[] = {m.rows(), m.cols()};
    npy_intp const stride[] = {m.rowStride() * 8, m.colStride() * 8};
    int ndim = 2;
    auto *m_npy = (PyArrayObject*) PyArray_New(&PyArray_Type, ndim, dims, NPY_FLOAT64,
                                               stride, m.data(), -1, 0, NULL);
    PyObject *m_npy_base = PyCapsule_New(m.data(), NULL, destruct_capsule);
    PyArray_SetBaseObject(m_npy, m_npy_base);
    return m_npy;
}


double pl_dist(vector<double> point, vector<double> start, vector<double> end) {
    bool equal = true;
    for (size_t i = 0; i < start.size(); i++) {
        if (start[i] != end[i]) {
            equal = false;
            break;
        }
    }

    if (equal) {
        double sum = 0.0;
        for (size_t i = 0; i < point.size(); i++) {
            double diff = point[i] - start[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }

    vector<double> v1(3), v2(3);
    for (size_t i = 0; i < 3; i++) {
        v1[i] = end[i] - start[i];
        v2[i] = start[i] - point[i];
    }

    vector<double> cross(3);
    cross[0] = v1[1] * v2[2] - v1[2] * v2[1];
    cross[1] = v1[2] * v2[0] - v1[0] * v2[2];
    cross[2] = v1[0] * v2[1] - v1[1] * v2[0];

    double cross_norm = 0.0;
    for (size_t i = 0; i < 3; i++) {
        cross_norm += cross[i] * cross[i];
    }
    cross_norm = sqrt(cross_norm);

    double line_norm = 0.0;
    for (size_t i = 0; i < v1.size(); i++) {
        line_norm += v1[i] * v1[i];
    }
    line_norm = sqrt(line_norm);

    return abs(cross_norm) / line_norm;
}


// Helper function to convert MapMatrix row to vector<double>
vector<double> get_row(const MapMatrix& M, E::Index idx) {
    vector<double> row(M.cols());
    for (E::Index i = 0; i < M.cols(); i++) {
        row[i] = M(idx, i);
    }
    return row;
}


// Recursive RDP implementation
MapMatrix rdp_rec(const MapMatrix& M, double epsilon) {
    if (M.rows() <= 2) {
        MapMatrix result = new_mat(M.rows(), M.cols());
        result = M;
        return result;
    }

    double dmax = 0.0;
    E::Index index = -1;

    vector<double> start(M.cols());
    vector<double> end(M.cols());
    for (E::Index i = 0; i < M.cols(); i++) {
        start[i] = M(0, i);
        end[i] = M(M.rows() - 1, i);
    }

    for (E::Index i = 1; i < M.rows(); i++) {
        vector<double> point(M.cols());
        for (E::Index j = 0; j < M.cols(); j++) {
            point[j] = M(i, j);
        }
        double d = pl_dist(point, start, end);
        if (d > dmax) {
            index = i;
            dmax = d;
        }
    }

    if (dmax > epsilon) {
        // Materialize the blocks to avoid deep template nesting
        auto _M1 = M.topRows(index + 1);
        MapMatrix M1 = new_mat(index + 1, M.cols());
        M1 = _M1;

        auto _M2 = M.bottomRows(M.rows() - index);
        MapMatrix M2 = new_mat(M.rows() - index, M.cols());
        M2 = _M2;

        MapMatrix r1 = rdp_rec(M1, epsilon);
        MapMatrix r2 = rdp_rec(M2, epsilon);

        MapMatrix result = new_mat(r1.rows() + r2.rows() - 1, M.cols());
        result.topRows(r1.rows()) = r1;
        result.bottomRows(r2.rows() - 1) = r2.bottomRows(r2.rows() - 1);

        del_mat(M1);
        del_mat(M2);
        del_mat(r1);
        del_mat(r2);

        return result;
    } else {
        MapMatrix result = new_mat(2, M.cols());
        result.row(0) = M.row(0);
        result.row(1) = M.row(M.rows() - 1);
        return result;
    }
}


// Helper for iterative RDP
vector<bool> _rdp_iter(const MapMatrix& M, E::Index start_index, E::Index last_index, double epsilon) {
    vector<std::pair<E::Index, E::Index>> stk;
    stk.push_back({start_index, last_index});

    E::Index global_start_index = start_index;
    vector<bool> indices(last_index - start_index + 1, true);

    while (!stk.empty()) {
        auto [si, li] = stk.back();
        stk.pop_back();

        double dmax = 0.0;
        E::Index index = si;

        vector<double> start = get_row(M, si);
        vector<double> end = get_row(M, li);

        for (E::Index i = si + 1; i < li; i++) {
            if (indices[i - global_start_index]) {
                vector<double> point = get_row(M, i);
                double d = pl_dist(point, start, end);
                if (d > dmax) {
                    index = i;
                    dmax = d;
                }
            }
        }

        if (dmax > epsilon) {
            stk.push_back({si, index});
            stk.push_back({index, li});
        } else {
            for (E::Index i = si + 1; i < li; i++) {
                indices[i - global_start_index] = false;
            }
        }
    }

    return indices;
}


// Iterative RDP that returns mask
MapMatrix rdp_iter_mask(const MapMatrix& M, double epsilon) {
    vector<bool> mask = _rdp_iter(M, 0, M.rows() - 1, epsilon);

    // Count true values
    E::Index count = 0;
    for (bool b : mask) {
        if (b) count++;
    }

    MapMatrix result = new_mat(count, M.cols());
    E::Index idx = 0;
    for (size_t i = 0; i < mask.size(); i++) {
        if (mask[i]) {
            result.row(idx++) = M.row(i);
        }
    }

    return result;
}


// Iterative RDP that returns boolean mask as matrix
MapMatrix rdp_iter_return_mask(const MapMatrix& M, double epsilon) {
    vector<bool> mask = _rdp_iter(M, 0, M.rows() - 1, epsilon);

    MapMatrix result = new_mat(mask.size(), 1);
    for (size_t i = 0; i < mask.size(); i++) {
        result(i, 0) = mask[i] ? 1.0 : 0.0;
    }

    return result;
}


MatOrStr rdp(const MapMatrix& x, double eps, const string& algo, bool return_mask) {
    if (algo == "iter") {
        if (return_mask) {
            return {rdp_iter_return_mask(x, eps)};
        } else {
            return {rdp_iter_mask(x, eps)};
        }
    } else if (algo == "rec") {
        if (return_mask) {
            const string err = "return_mask=True not supported with algo=\"rec\"";
            return {err};
        }
        return {rdp_rec(x, eps)};
    } else {
        const string err = std::format("Invalid algorithm '{}'. Must be either 'iter' or 'rec'.", algo);
        return {err};
    }
}


static PyObject *py_rdp(PyObject *self, PyObject *args) {
    PyArrayObject *x;
    double eps;
    PyObject *dist_func; // Currently unused
    const char *_algo;
    int _return_mask;
    int success = PyArg_ParseTuple(args, "O!dOsp:rdp", &PyArray_Type, &x, &eps,
                                   &dist_func, &_algo, &_return_mask);
    if (!success) {
        PyErr_SetString(PyExc_Exception, "Bad Arguments!!");
        return NULL;
    }
    MatOrStr _x_mat = to_matrix(x);
    if (!_x_mat.is_mat()) {
        string& err = std::get<string>(_x_mat.data);
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    auto x_mat = std::move(std::get<MapMatrix>(_x_mat.data));
    const string algo(_algo);
    bool return_mask = _return_mask;

    MatOrStr _res = rdp(x_mat, eps, algo, return_mask);
    if (!_res.is_mat()) {
        string& err = std::get<string>(_res.data);
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    auto res = std::move(std::get<MapMatrix>(_res.data));
    PyArrayObject *res_npy = to_ndarray(res);
    return (PyObject*) res_npy;
}


static PyMethodDef methods[] = {
    {"rdp", py_rdp, METH_VARARGS, PyDoc_STR("Run RDP algorithm on a sequence.")},
    {NULL, NULL, 0, NULL},
};


static struct PyModuleDef module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "fastrdp._core",
    .m_doc = PyDoc_STR("C++ implementation of RDP algorithm."),
    .m_methods = methods,
};


PyMODINIT_FUNC PyInit__core(void) {
    import_array();
    return PyModuleDef_Init(&module);
}
