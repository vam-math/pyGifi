#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* =========================================================================
 * 1. Exact MT19937 Implementation (from R's RNG.c) 
 * ========================================================================= */
#define N_MT 624
#define M_MT 397
#define MATRIX_A 0x9908b0df
#define UPPER_MASK 0x80000000
#define LOWER_MASK 0x7fffffff

#define TEMPERING_MASK_B 0x9d2c5680
#define TEMPERING_MASK_C 0xefc60000
#define TEMPERING_SHIFT_U(y)  (y >> 11)
#define TEMPERING_SHIFT_S(y)  (y << 7)
#define TEMPERING_SHIFT_T(y)  (y << 15)
#define TEMPERING_SHIFT_L(y)  (y >> 18)

static uint32_t mt[N_MT];
static int mti = N_MT + 1;

static void MT_sgenrand(uint32_t seed) {
    int i;
    for (i = 0; i < N_MT; i++) {
        mt[i] = seed & 0xffff0000;
        seed = 69069 * seed + 1;
        mt[i] |= (seed & 0xffff0000) >> 16;
        seed = 69069 * seed + 1;
    }
    mti = N_MT;
}

static double MT_genrand(void) {
    uint32_t y;
    static uint32_t mag01[2] = {0x0, MATRIX_A};

    if (mti >= N_MT) { 
        int kk;
        if (mti == N_MT + 1) MT_sgenrand(4357);
        for (kk = 0; kk < N_MT - M_MT; kk++) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
            mt[kk] = mt[kk+M_MT] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        for (; kk < N_MT - 1; kk++) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
            mt[kk] = mt[kk+(M_MT-N_MT)] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        y = (mt[N_MT-1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N_MT-1] = mt[M_MT-1] ^ (y >> 1) ^ mag01[y & 0x1];
        mti = 0;
    }

    y = mt[mti++];
    y ^= TEMPERING_SHIFT_U(y);
    y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
    y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
    y ^= TEMPERING_SHIFT_L(y);

    return ((double)y * 2.3283064365386963e-10); /* [0,1) */
}

#define i2_32m1 2.328306437080797e-10
static double fixup(double x)
{
    /* ensure 0 and 1 are never returned */
    if(x <= 0.0) return 0.5*i2_32m1;
    if((1.0 - x) <= 0.0) return 1.0 - 0.5*i2_32m1;
    return x;
}

static double unif_rand(void) {
    return fixup(MT_genrand());
}

/* =========================================================================
 * 2. Inversion normal generation AS241 (from R's qnorm.c)
 * ========================================================================= */
static double qnorm_as241(double p) {
    double q, r, val;

    if (p == 0.5) return 0.0;
    
    q = p - 0.5;

    if (fabs(q) <= .425) { 
        r = .180625 - q * q; 
        val = q * (((((((r * 2509.0809287301226727 +
                       33430.575583588128105) * r + 67265.770927008700853) * r +
                     45921.953931549871457) * r + 13731.693765509461125) * r +
                   1971.5909503065514427) * r + 133.14166789178437745) * r +
                 3.387132872796366608)
            / (((((((r * 5226.495278852854561 +
                     28729.085735721942674) * r + 39307.89580009271061) * r +
                   21213.794301586595867) * r + 5394.1960214247511077) * r +
                 687.1870074920579083) * r + 42.313330701600911252) * r + 1.);
    }
    else { 
        double lp = (q > 0) ? log(1.0 - p) : log(p); 
        r = sqrt(-lp);
        if (r <= 5.) {
            r += -1.6;
            val = (((((((r * 7.7454501427834140764e-4 +
                       .0227238449892691845833) * r + .24178072517745061177) *
                     r + 1.27045825245236838258) * r +
                    3.64784832476320460504) * r + 5.7694972214606914055) *
                  r + 4.6303378461565452959) * r +
                 1.42343711074968357734)
                / (((((((r *
                         1.05075007164441684324e-9 + 5.475938084995344946e-4) *
                        r + .0151986665636164571966) * r +
                       .14810397642748007459) * r + .68976733498510000455) *
                     r + 1.6763848301838038494) * r +
                    2.05319162663775882187) * r + 1.);
        }
        else if(r <= 27) { 
            r += -5.;
            val = (((((((r * 2.01033439929228813265e-7 +
                       2.71155556874348757815e-5) * r +
                      .0012426609473880784386) * r + .026532189526576123093) *
                    r + .29656057182850489123) * r +
                   1.7848265399172913358) * r + 5.4637849111641143699) *
                 r + 6.6579046435011037772)
                / (((((((r *
                         2.04426310338993978564e-15 + 1.4215117583164458887e-7)*
                        r + 1.8463183175100546818e-5) * r +
                       7.868691311456132591e-4) * r + .0148753612908506148525)
                     * r + .13692988092273580531) * r +
                    .59983220655588793769) * r + 1.);
        }
        else {
            if(r >= 6.4e8) {
                val = r * 1.4142135623730950488016887242097; /* sqrt(2) */
            } else {
                double s2 = -ldexp(lp, 1);
                double x2 = s2 - log(6.283185307179586476925286766559 * s2); /* 2*pi */
                if(r < 36000.) {
                    x2 = s2 - log(6.283185307179586476925286766559 * x2) - 2./(2. + x2); 
                    if(r < 840.) { 
                        x2 = s2 - log(6.283185307179586476925286766559 * x2) + 2*log1p(- (1 - 1/(4 + x2))/(2. + x2)); 
                        if(r < 109.) { 
                          x2 = s2 - log(6.283185307179586476925286766559 * x2) +
                              2*log1p(- (1 - (1 - 5/(6 + x2))/(4. + x2))/(2. + x2)); 
                          if(r < 55.) { 
                            x2 = s2 - log(6.283185307179586476925286766559 * x2) +
                              2*log1p(- (1 - (1 - (5 - 9/(8. + x2))/(6. + x2))/(4. + x2))/(2. + x2)); 
                          }
                        }
                    }
                }
                val = sqrt(x2);
            }
        }
        if(q < 0.0) val = -val;
    }
    return val;
}

/* =========================================================================
 * 3. Python C-API Interface
 * ========================================================================= */

static void do_r_set_seed(unsigned int seed) {
    int i;
    for(i = 0; i < 50; i++) seed = (69069 * seed + 1);

    uint32_t iseed[N_MT+1];  
    for(i = 0; i <= N_MT; i++) {
        seed = (69069 * seed + 1);
        iseed[i] = seed;
    }
    
    iseed[0] = 624;

    for(i=0; i < N_MT; i++) {
        mt[i] = iseed[i+1];
    }
    mti = 624;
}

static PyObject* py_r_set_seed(PyObject* self, PyObject* args) {
    unsigned int seed;
    if (!PyArg_ParseTuple(args, "I", &seed)) return NULL;
    do_r_set_seed(seed);
    Py_RETURN_NONE;
}

#define BIG 134217728.0 /* 2^27 */

static double norm_rand_custom(void) {
    /* Exact R 53-bit inversion construction!
       See snorm.c:266
       u1 = unif_rand();
       u1 = (int)(BIG*u1) + unif_rand();
       return qnorm(u1/BIG, 0.0, 1.0, 1, 0); 
    */
    double u1 = unif_rand();
    u1 = (int)(BIG * u1) + unif_rand();
    return qnorm_as241(u1 / BIG);
}

static PyObject* py_r_rnorm(PyObject* self, PyObject* args) {
    int n;
    if (!PyArg_ParseTuple(args, "i", &n)) return NULL;

    PyObject* result = PyList_New(n);
    for (int i = 0; i < n; i++) {
        double val = norm_rand_custom();
        PyList_SET_ITEM(result, i, PyFloat_FromDouble(val));
    }
    return result;
}

static PyObject* py_r_init_x(PyObject* self, PyObject* args) {
    int n, p;
    unsigned int seed;
    if (!PyArg_ParseTuple(args, "iiI", &n, &p, &seed)) return NULL;
    do_r_set_seed(seed);

    double* mat = (double*)malloc(n * p * sizeof(double));
    for (int j = 0; j < p; j++) {
        for (int i = 0; i < n; i++) mat[i + j * n] = norm_rand_custom();
    }

    PyObject* result = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyObject* row = PyList_New(p);
        for (int j = 0; j < p; j++) PyList_SET_ITEM(row, j, PyFloat_FromDouble(mat[i + j * n]));
        PyList_SET_ITEM(result, i, row);
    }
    free(mat);
    return result;
}

static PyMethodDef RngMethods[] = {
    {"r_set_seed", py_r_set_seed, METH_VARARGS, "Set R-compatible MT19937 seed"},
    {"r_rnorm",    py_r_rnorm,    METH_VARARGS, "Generate n normal deviates (Inversion+AS241)"},
    {"r_init_x",   py_r_init_x,  METH_VARARGS, "Generate R-compatible n x p init matrix"},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef rngmodule = { PyModuleDef_HEAD_INIT, "pygifi_rng", NULL, -1, RngMethods };
PyMODINIT_FUNC PyInit_pygifi_rng(void) { return PyModule_Create(&rngmodule); }
