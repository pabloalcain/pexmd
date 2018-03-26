void periodic(float *x, long int npart, float *x0, float *xf) {
  for (int i = 0; i < npart; i++) {
    for (int k = 0; k < 3; k ++) {
      while (x[3*i + k] > xf[k]) x[3*i + k] -= (xf[k] - x0[k]);
      while (x[3*i + k] < x0[k]) x[3*i + k] += (xf[k] - x0[k]);
    }
  }
  return;
}

void fixed(float *x, float *v, long int npart, float *x0, float *xf) {
  for (int i = 0; i < npart; i++) {
    for (int k = 0; k < 3; k ++) {
      while ((x[3*i + k] > xf[k]) || (x[3*i + k] < x0[k])) {
        v[3*i + k] *= -1;
        if (x[3*i + k] > xf[k]) {
          x[3*i + k] = 2*xf[k] - x[3*i + k];
        }
        else {
          x[3*i + k] = 2*x0[k] - x[3*i + k];
        }
      }
    }
  }
  return;
}
