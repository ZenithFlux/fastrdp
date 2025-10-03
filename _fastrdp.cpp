#include "Eigen/Dense"
#include <format>
#include <string>
#include <variant>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/ndarrayobject.h>


namespace E = Eigen;
using std::string;
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


MatOrStr to_matrix(PyArrayObject* x) {
    int x_ndim = PyArray_NDIM(x);
    if (x_ndim != 2) {
        const string err = std::format("The array should have 2 dims, but has {} dims.", x_ndim);
        return MatOrStr(err);
    }
    char& x_type = PyArray_DESCR(x)->type;
    if (x_type != 'd') {
        const string err = std::format("The array should have type d, but has type {}.", x_type);
        return MatOrStr(err);
    }

    npy_intp r = PyArray_DIM(x, 0), c = PyArray_DIM(x, 1);
    npy_intp sr = PyArray_STRIDE(x, 0), sc = PyArray_STRIDE(x, 1);
    double *x_data = static_cast<double *>(PyArray_DATA(x));
    const DStride x_mat_s(sr / 8, sc / 8);
    MapMatrix x_mat(x_data, r, c, x_mat_s);
    return MatOrStr(x_mat);
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


static PyObject *py_rdp(PyObject *self, PyObject *args) {
    PyArrayObject *x;
    double eps;
    PyObject *dist_func;
    const char *_algo;
    int _return_mask;
    int success = PyArg_ParseTuple(args, "O!dOsp:rdp", &PyArray_Type, &x, &eps,
                                   &dist_func, &_algo, &_return_mask);
    if (!success || !PyCallable_Check(dist_func)) {
        return NULL;
    }
    MatOrStr _x_mat = to_matrix(x);
    if (!_x_mat.is_mat()) {
        string& err = std::get<string>(_x_mat.data);
        PyErr_SetString(PyExc_ValueError, err.c_str());
        return NULL;
    }
    MapMatrix x_mat = std::move(std::get<MapMatrix>(_x_mat.data));
    const string algo(_algo);
    bool return_mask = _return_mask;

    // Dummy operations
    E::Index r = x_mat.rows(), c = x_mat.cols();
    double *res_data = new double[r*c];
    MapMatrix res(res_data, r, c, DStride(c, 1));
    res = x_mat;
    res *= 2;
    //

    PyArrayObject *res_npy = to_ndarray(res);
    return (PyObject*) res_npy;
}


static PyMethodDef methods[] = {
    {"rdp", py_rdp, METH_VARARGS, PyDoc_STR("Run RDP algorithm on a sequence.")},
    {NULL, NULL, 0, NULL},
};


static struct PyModuleDef module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_fastrdp",
    .m_doc = PyDoc_STR("C++ implementation of RDP algorithm."),
    .m_methods = methods,
};


PyMODINIT_FUNC PyInit__fastrdp(void) {
    import_array();
    return PyModuleDef_Init(&module);
}
