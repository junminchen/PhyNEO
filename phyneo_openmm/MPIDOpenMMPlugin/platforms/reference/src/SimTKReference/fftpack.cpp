#include "fftpack.h"

#include <cmath>
#include <complex>
#include <vector>

struct fftpack {
    int nx = 0;
    int ny = 1;
    int nz = 1;
    int dim = 1;
};

static inline bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static inline std::complex<double> to_complex(const t_complex& x) {
    return std::complex<double>(x.re, x.im);
}

static inline t_complex to_tcomplex(const std::complex<double>& x) {
    return t_complex(x.real(), x.imag());
}

// Iterative radix-2 FFT (unnormalized), falls back to O(N^2) DFT for non power-of-two sizes.
static void fft_1d(const std::complex<double>* in, std::complex<double>* out, int n, bool forward) {
    if (!is_power_of_two(n)) {
        const double sign = forward ? -1.0 : 1.0;
        const double twopi = 2.0 * M_PI;
        for (int k = 0; k < n; ++k) {
            std::complex<double> sum(0.0, 0.0);
            for (int j = 0; j < n; ++j) {
                const double ang = sign * twopi * static_cast<double>(j) * static_cast<double>(k) / static_cast<double>(n);
                const std::complex<double> w(std::cos(ang), std::sin(ang));
                sum += in[j] * w;
            }
            out[k] = sum;
        }
        return;
    }

    for (int i = 0; i < n; ++i) {
        out[i] = in[i];
    }

    // bit-reversal permutation
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(out[i], out[j]);
        }
    }

    const double sign = forward ? -1.0 : 1.0;
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = sign * 2.0 * M_PI / static_cast<double>(len);
        const std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                const std::complex<double> u = out[i + j];
                const std::complex<double> v = out[i + j + len / 2] * w;
                out[i + j] = u + v;
                out[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

static int exec_nd(const fftpack* p, enum fftpack_direction dir, t_complex* in_data, t_complex* out_data) {
    if (p == nullptr || in_data == nullptr || out_data == nullptr) {
        return 1;
    }
    const bool forward = (dir == FFTPACK_FORWARD);
    const int nx = p->nx;
    const int ny = p->ny;
    const int nz = p->nz;
    const int n = nx * ny * nz;

    std::vector<std::complex<double>> grid(n);
    for (int i = 0; i < n; ++i) {
        grid[i] = to_complex(in_data[i]);
    }
    std::vector<std::complex<double>> tmp(std::max(nx, std::max(ny, nz)));
    std::vector<std::complex<double>> outv(std::max(nx, std::max(ny, nz)));

    // x dimension
    if (nx > 1) {
        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                const int base = (z * ny + y) * nx;
                for (int x = 0; x < nx; ++x) {
                    tmp[x] = grid[base + x];
                }
                fft_1d(tmp.data(), outv.data(), nx, forward);
                for (int x = 0; x < nx; ++x) {
                    grid[base + x] = outv[x];
                }
            }
        }
    }

    // y dimension
    if (ny > 1) {
        for (int z = 0; z < nz; ++z) {
            for (int x = 0; x < nx; ++x) {
                for (int y = 0; y < ny; ++y) {
                    tmp[y] = grid[(z * ny + y) * nx + x];
                }
                fft_1d(tmp.data(), outv.data(), ny, forward);
                for (int y = 0; y < ny; ++y) {
                    grid[(z * ny + y) * nx + x] = outv[y];
                }
            }
        }
    }

    // z dimension
    if (nz > 1) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                for (int z = 0; z < nz; ++z) {
                    tmp[z] = grid[(z * ny + y) * nx + x];
                }
                fft_1d(tmp.data(), outv.data(), nz, forward);
                for (int z = 0; z < nz; ++z) {
                    grid[(z * ny + y) * nx + x] = outv[z];
                }
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        out_data[i] = to_tcomplex(grid[i]);
    }
    return 0;
}

extern "C" {

int fftpack_init_1d(fftpack_t* fft, int nx) {
    if (fft == nullptr || nx <= 0) {
        return 1;
    }
    fftpack* p = new fftpack();
    p->nx = nx;
    p->ny = 1;
    p->nz = 1;
    p->dim = 1;
    *fft = p;
    return 0;
}

int fftpack_init_2d(fftpack_t* fft, int nx, int ny) {
    if (fft == nullptr || nx <= 0 || ny <= 0) {
        return 1;
    }
    fftpack* p = new fftpack();
    p->nx = nx;
    p->ny = ny;
    p->nz = 1;
    p->dim = 2;
    *fft = p;
    return 0;
}

int fftpack_init_3d(fftpack_t* fft, int nx, int ny, int nz) {
    if (fft == nullptr || nx <= 0 || ny <= 0 || nz <= 0) {
        return 1;
    }
    fftpack* p = new fftpack();
    p->nx = nx;
    p->ny = ny;
    p->nz = nz;
    p->dim = 3;
    *fft = p;
    return 0;
}

int fftpack_exec_1d(fftpack_t setup, enum fftpack_direction dir, t_complex* in_data, t_complex* out_data) {
    return exec_nd(setup, dir, in_data, out_data);
}

int fftpack_exec_2d(fftpack_t setup, enum fftpack_direction dir, t_complex* in_data, t_complex* out_data) {
    return exec_nd(setup, dir, in_data, out_data);
}

int fftpack_exec_3d(fftpack_t setup, enum fftpack_direction dir, t_complex* in_data, t_complex* out_data) {
    return exec_nd(setup, dir, in_data, out_data);
}

void fftpack_destroy(fftpack_t setup) {
    delete setup;
}

}  // extern "C"
