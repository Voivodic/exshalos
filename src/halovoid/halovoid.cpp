#define FINDER_MODULE

// Import the libraries with the functions of this module
#include "finder.hpp"
#include "halovoid_h.hpp"

// Import the headers with python and numpy APIs
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// This declares the compute function
static PyObject *halovoid_check_precision(PyObject *self, PyObject *args);
static PyObject *total_cell_volume(PyObject *self, PyObject *args, PyObject *kwargs);
//static PyObject *find(PyObject *self, PyObject *args, PyObject *kwargs);

/*This tells Python what methods this module has. See the Python-C API for more
 * information.*/
static PyMethodDef halovoid_methods[] = {
    {"check_precision", (PyCFunction)halovoid_check_precision, METH_VARARGS,
     "Returns precision used by the estimators of the spectra"},
    {"total_volume", (PyCFunction)total_cell_volume, METH_VARARGS | METH_KEYWORDS,
     "Compute the total volume of all cells."},
    // {"find", (PyCFunction)find, METH_VARARGS | METH_KEYWORDS,
    //  "Find halos and voids in 2D or 3D using voro++."},
    {NULL, NULL, 0, NULL}};

/*Return the precision used in the grid computations*/
static PyObject *halovoid_check_precision(PyObject *self, PyObject *args) {
    return Py_BuildValue("i", sizeof(fft_real));
}

// Function that computes the total volume of all cells
static PyObject *total_cell_volume(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
    size_t np, ndim;
    int Nd;
    fft_real *pos;
    fft_real L;

    // Define the list of parameters
    static const char *kwlist[] = {"particles", "Nd", "L", NULL};

    // Define the pyobjests
    PyArrayObject *pos_array;

// Read the input arguments
#ifdef DOUBLEPRECISION_FFTW
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oid", kwlist, &pos_array,
                                     &Nd, &L))
        return NULL;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oif", kwlist, &pos_array,
                                     &Nd, &L))
        return NULL;
#endif

    /*Convert the PyObjects to C arrays*/
    pos = static_cast<fft_real *>(PyArray_DATA(pos_array));
    np = (size_t)PyArray_DIMS(pos_array)[0];
    ndim = (size_t)PyArray_DIMS(pos_array)[1];

    // Check the dimensionality of the input array
    if (ndim != 3) { // Only 3D arrays are supported
        PyErr_SetString(PyExc_ValueError, "Only 3D arrays are supported");
        return NULL;
    }

    // Check the number of particles
    if (np < 1) {
        PyErr_SetString(PyExc_ValueError, "At least one particle is required");
        return NULL;
    }

    // Check the number of dimensions
    if (Nd < 1) {
        PyErr_SetString(PyExc_ValueError, "The number of dimensions must be at "
                                          "least 1");
        return NULL;
    }

    // Check the length of the box
    if (L < 1) {
        PyErr_SetString(PyExc_ValueError, "The length of the box must be at "
                                          "least 1");
        return NULL;
    }

    // Create the container
    size_t *offset = new size_t[Nd * Nd * Nd + 1];
    create_container(pos, offset, np, Nd, L);

    // Compute the total volume
    double volume = total_volume(pos, offset, np, Nd, L);
    delete[] offset;

    return Py_BuildValue("d", volume);
}

// Function that finds halos and voids in 2D or 3D using voro++
// static PyObject *find(PyObject *self, PyObject *args, PyObject *kwargs) {
//     size_t np, ndim, n_halos, n_voids;
//     int find_halos, find_voids, nd;
//     fft_real *pos, *halos, *voids;
//     fft_real delta_h, delta_v, L, r_max;
// 
//     // Define the list of parameters
//     static const char *kwlist[] = {"particles", "find_halos", "find_voids",
//                              "delta_h",   "delta_v",    "L",
//                              "nd",        "r_max",      NULL};
// 
//     // Define the pyobjests
//     PyArrayObject *pos_array;
// 
// // Read the input arguments
// #ifdef DOUBLEPRECISION_FFTW
//     if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiidddid", kwlist,
//                                      &pos_array, &find_halos, &find_voids,
//                                      &delta_h, &delta_v, &L, &nd, &r_max))
//         return NULL;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiifffif", kwlist,
//                                      &pos_array, &find_halos, &find_voids,
//                                      &delta_h, &delta_v, &L, &nd, &r_max))
//         return NULL;
// #endif
// 
//     /*Convert the PyObjects to C arrays*/
//     pos = (fft_real *)PyArray_DATA(pos_array);
//     np = (size_t)PyArray_DIMS(pos_array)[0];
//     ndim = (size_t)PyArray_DIMS(pos_array)[1];
// 
//     // Find the 2D halos and voids
//     n_halos = 0;
//     n_voids = 0;
//     find_halos_voids(pos, np, find_halos, find_voids, delta_h, delta_v, L, nd,
//                      r_max, &halos, &n_halos, &voids, &n_voids);
// 
//     // Create the PyObjects for the output
//     PyObject *dict = PyDict_New();
//     if (dict == NULL)
//         return NULL;
//     if (n_halos > 0) {
//         npy_intp dims_halos[] = {(npy_intp)n_halos, (npy_intp)ndim + 1};
//         PyObject *np_halos =
//             PyArray_SimpleNewFromData(2, dims_halos, NP_OUT_TYPE, halos);
//         PyArray_ENABLEFLAGS((PyArrayObject *)np_halos, NPY_ARRAY_OWNDATA);
//         PyDict_SetItemString(dict, "halos", np_halos);
//         Py_DECREF(np_halos);
//     }
//     if (n_voids > 0) {
//         npy_intp dims_voids[] = {(npy_intp)n_voids, (npy_intp)ndim + 1};
//         PyObject *np_voids =
//             PyArray_SimpleNewFromData(2, dims_voids, NP_OUT_TYPE, voids);
//         PyArray_ENABLEFLAGS((PyArrayObject *)np_voids, NPY_ARRAY_OWNDATA);
//         PyDict_SetItemString(dict, "voids", np_voids);
//         Py_DECREF(np_voids);
//     }
// 
//     return dict;
// }

#ifdef __cplusplus
extern "C" {
#endif

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef halovoid_module = {PyModuleDef_HEAD_INIT, "halovoid",
                                             NULL, -1, halovoid_methods};
PyMODINIT_FUNC PyInit_halovoid(void) {
    import_array();
    return PyModule_Create(&halovoid_module);
}
#else
PyMODINIT_FUNC inithalovoid(void) {
    import_array();
    Py_InitModule("halovoid", halovoid_methods);
}
#endif

#ifdef __cplusplus
}
#endif
