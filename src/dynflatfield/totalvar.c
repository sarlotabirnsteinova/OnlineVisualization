#include <math.h>

double totalvar(
  int ny, int nx, int nc, double *w, float *img,
  float *flat0, float flat0_mean, float *comp, float *comp_mean,
  float *wrkspc)
{
  int x, y, k, yx, yxk, npx;
  float v, vx, factor, flat, dx, dy;
  float *vy;
  double I;
  
  npx = ny * nx;
  vy = wrkspc;
  vx = 0.0;

  factor = flat0_mean;
  for (k = 0; k < nc; k++)
    factor += comp_mean[k] * (float) w[k];
    
  I = 0.0;
  yx = 0;
  for (y = 0; y < ny; y++)
    for (x = 0; x < nx; x++) {
      flat = flat0[yx];
      yxk = yx;
      for (k = 0; k < nc; k++) {
        flat += comp[yxk] * (float) w[k];
        yxk += npx;
      }
        //flat += comp[yxk++] * (float) w[k];

      v = img[yx] / flat * factor;

      dx = (x > 0) ? (vx - v) : 0.0;
      dy = (y > 0) ? (vy[x] - v) : 0.0;

      I += sqrtf(dx * dx + dy * dy);

      vy[x] = vx = v;
      yx++;
  }
  return I;
}

void grad_totalvar(
  int ny, int nx, int nc, double *w, float *img,
  float *flat0, float flat0_mean, float *comp, float *comp_mean,
  float *wrkspc, double *J)
{
  int x, y, k, xk, yx, yxk, npx;
  float v, vx, factor, flat, dx, dy;
  float u, g, f, dv, dxdw, dydw, factor_flat;
  float *vy, *dvx, *dvy;
  
  npx = ny * nx;
  vy = wrkspc;
  dvx = vy + nx;
  dvy = dvx + nc;
  vx = 0.0;

  factor = flat0_mean;
  for (k = 0; k < nc; k++) {
    factor += comp_mean[k] * (float) w[k];
    J[k] = 0.0;
  }

  yx = 0;
  for (y = 0; y < ny; y++) {
    xk = 0;
    for (x = 0; x < nx; x++) {
      flat = flat0[yx];
      yxk = yx;
      for (k = 0; k < nc; k++) {
        flat += comp[yxk] * (float) w[k];
        yxk += npx;  
      }
      u = img[yx] / flat;
      v = u * factor;

      dx = (x > 0) ? (vx - v) : 0.0;
      dy = (y > 0) ? (vy[x] - v) : 0.0;

      f = sqrtf(dx * dx + dy * dy);
      factor_flat = factor / flat;
      yxk = yx;
      for (k = 0; k < nc; k++) {
        dv = u * (comp_mean[k] - factor_flat * comp[yxk]);
        yxk += npx;

        dxdw = (x > 0) ? (dvx[k] - dv) : 0.0;
        dydw = (y > 0) ? (dvy[xk] - dv) : 0.0;
          
        g = dx * dxdw + dy * dydw;
          
        J[k] += (f != 0.0) ? (g / f) : 0.0;
          
        dvy[xk] = dvx[k] = dv;
        xk++;
      }

      vy[x] = vx = v;
      yx++;
    }
  }
}
