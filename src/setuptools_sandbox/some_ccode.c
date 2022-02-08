#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject* addfloats(PyObject* self, PyObject* args) {
	double arg1, arg2;
	if (!PyArg_ParseTuple(args, "dd", &arg1, &arg2)) {
		return NULL;
	}

	double sum = arg1 + arg2;

	return PyFloat_FromDouble(sum);
}


struct module_state {
	PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static PyMethodDef addfloats_methods[] = {
	{"addfloats", addfloats, METH_VARARGS, "Add 2 floats"},
	{NULL, NULL, 0, NULL}
};

static int addfloats_traverse(PyObject *m, visitproc visit, void *arg) {
	Py_VISIT(GETSTATE(m)->error);
	return 0;
}

static int addfloats_clear(PyObject *m) {
	Py_CLEAR(GETSTATE(m)->error);
	return 0;
}


static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT,
	"addfloats",
	NULL,
	sizeof(struct module_state),
	addfloats_methods,
	NULL,
	addfloats_traverse,
	addfloats_clear,
	NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_addfloats(void)
{
	PyObject *module = PyModule_Create(&moduledef);

	if (module == NULL)
		INITERROR;
	struct module_state *st = GETSTATE(module);

	st->error = PyErr_NewException("addfloats.Error", NULL, NULL);
	if (st->error == NULL) {
		Py_DECREF(module);
		INITERROR;
	}

	/* IMPORTANT: this must be called for NumPy */
	import_array();

	return module;
}


