const char* dgemm_desc = "Blocked dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{
    int Nb = n / block_size;

    double* blockA = new double[block_size * block_size];
    double* blockB = new double[block_size * block_size];
    double* blockC = new double[block_size * block_size];

    for (int i = 0; i < Nb; i++) {
        for (int j = 0; j < Nb; j++) {
            for (int ii = 0; ii < block_size; ii++) {
                for (int jj = 0; jj < block_size; jj++) {
                    blockC[ii + jj * block_size] = C[(i*block_size + ii) + (j*block_size + jj) * n];
                }
            }

            for (int k = 0; k < Nb; k++) {
                for (int ii = 0; ii < block_size; ii++) {
                    for (int kk = 0; kk < block_size; kk++) {
                        blockA[ii + kk * block_size] = A[(i*block_size + ii) + (k*block_size + kk) * n];
                    }
                }

                for (int kk = 0; kk < block_size; kk++) {
                    for (int jj = 0; jj < block_size; jj++) {
                        blockB[kk + jj * block_size] = B[(k*block_size + kk) + (j*block_size + jj) * n];
                    }
                }

                for (int ii = 0; ii < block_size; ii++) {
                    for (int jj = 0; jj < block_size; jj++) {
                        for (int kk = 0; kk < block_size; kk++) {
                            blockC[ii + jj * block_size] += blockA[ii + kk * block_size] * blockB[kk + jj * block_size];
                        }
                    }
                }
            }

            for (int ii = 0; ii < block_size; ii++) {
                for (int jj = 0; jj < block_size; jj++) {
                    C[(i*block_size + ii) + (j*block_size + jj) * n] = blockC[ii + jj * block_size];
                }
            }
        }
    }

    // Cleanup
    delete[] blockA;
    delete[] blockB;
    delete[] blockC;
}