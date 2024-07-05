#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <fstream>

using namespace Eigen;
using namespace std;

// Const
int p =10;
int f = 750;
//

Matrix2d A_k(int k) {
    double angle = 0.2 * M_PI * k;
    Matrix2d output;
    output << 1, 0.2 * sin(angle),
             -0.1 * cos(angle), 0.9;
    return output;
}

Vector2d B_k(int k) {
    double angle = 0.2 * M_PI * k;
    Vector2d output;
    output << 0, 0.1 * cos(angle);
    return output;
}

void find_AB(MatrixXd &A,MatrixXd &B,MatrixXd &A1,MatrixXd &A2,MatrixXd &B1,MatrixXd &B2) {
    int n = B_k(0).rows();
    int m = B_k(0).cols();
    MatrixXd M = MatrixXd::Identity(n, n);
    MatrixXd F = MatrixXd::Zero(n*p, n);
    for(int i=0;i<p;i++) {
        M=A_k(i)*M;
        F.block<2, 2>(i*n,0)=M;
    }
    A <<  MatrixXd::Zero(p * n, (p - 1) * n),F;
    A1 << F.block(0,0,(p-1)*n,n);
    A2 << F.block((p-1)*n,0,n,n);
    MatrixXd C = MatrixXd::Zero(n*p, m);
    MatrixXd tmp;
    MatrixXd C0;
    for (int j=0;j<p;j++) {
        tmp = MatrixXd::Identity(n, n);
        for (int k=0;k<p;k++) {
            if(k<j) {
            C.block(k*n,0,n,m) << MatrixXd::Zero(n, m);
        }
            else if (k==j) {
                C0 =  B_k(k);
                C.block(k*n,0,n,m) << C0;
            }
            else {
                tmp = A_k(k)*tmp;
                C.block(k*n,0,n,m) << tmp*C0;
            }
        }
        B.block(0,j,p*n,m)=C;
    }
    B1 << B.block(0,0,(p-1)*n,p*m);
    B2 << B.block((p-1)*n,0,n,p*m);
}

MatrixXd randn(int rows, int cols) {
    // Khởi tạo generator với seed từ hệ thống
    random_device rd;
    mt19937 gen(rd());
    // Phân phối chuẩn với mean = 0 và stddev = 1
    normal_distribution<> d(0, 1);

    MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat(i, j) = d(gen);
        }
    }
    return mat;
}

void find_QR(MatrixXd &Q,MatrixXd &R,MatrixXd &Q1,MatrixXd &Q2,MatrixXd &R1,MatrixXd &R2, MatrixXd Q0,MatrixXd R0) {
    int n = B_k(0).rows();
    int m = B_k(0).cols();
    for (int i=0;i<p;i++) {
        Q.block(i*n,i*n,n,n) = Q0;
    }
    for (int j=0;j<p;j++) {
        R.block(j*m,j*m,m,m) = R0;
    }
    Q1 << Q.block(0,0,(p-1)*n,(p-1)*n);
    Q2 << Q.block((p-1)*n,(p-1)*n,n,n);
    R1 << R.block(0,0,(p-1)*m,(p-1)*m);
    R2 << R.block((p-1)*m,(p-1)*m,m,m);
}

void colect_data(MatrixXd &x,MatrixXd &x1,MatrixXd &x2,MatrixXd &u,MatrixXd A,MatrixXd B,MatrixXd K0) {
    int n = B_k(0).rows();
    int m = B_k(0).cols();
    for (int k=0;k<f;k++) {
        u.block(0,k,p*m,1) = -K0*x.block(0,k,p*n,1)+0.3*randn(p*m,1);
        x.block(0,k+1,p*n,1) = A*x.block(0,k,p*n,1)+ B*u.block(0,k,p*m,1);
        x1.block(0,k+1,(p-1)*n,1) = x.block(0,k+1,(p-1)*n,1);
        x2.block(0,k+1,n,1) = x.block((p-1)*n,k+1,n,1);
    }
}

VectorXd vecv(const MatrixXd& a) {
    // Tính ma trận m = a * a'
    MatrixXd m = a * a.transpose();

    // Khởi tạo vector p
    vector<double> p;
    int s = m.rows();

    // Lấy các phần tử trong tam giác trên của m
    for (int i = 0; i < s; ++i) {
        for (int j = i; j < s; ++j) {
            p.push_back(m(i, j));
        }
    }

    VectorXd y = Map<VectorXd>(p.data(), p.size());
    return y;
}

MatrixXd vecs_inv(const VectorXd& v) {
    int s = v.size();
    int n = static_cast<int>(0.5 * (-1 + sqrt(1 + 8 * s)));

    // Khởi tạo ma trận O với tam giác dưới là các phần tử từ vector v
    MatrixXd O = MatrixXd::Zero(n, n);
    int index = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            O(j, i) = v(index++);
        }
    }

    // Tính toán ma trận đối xứng y
    MatrixXd y = 0.5 * (O.transpose() + O);
    return y;
}

MatrixXd kroneckerProduct(const MatrixXd& A, const MatrixXd& B) {
    int rowsA = A.rows();
    int colsA = A.cols();
    int rowsB = B.rows();
    int colsB = B.cols();

    // Ma trận kết quả có kích thước (rowsA * rowsB) x (colsA * colsB)
    MatrixXd result(rowsA * rowsB, colsA * colsB);

    // Tính tích Kronecker
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            result.block(i * rowsB, j * colsB, rowsB, colsB) = A(i, j) * B;
        }
    }

    return result;
}

MatrixXd reshape(const MatrixXd& A, int rows, int cols) {
    // Đảm bảo rằng tổng số phần tử khớp với kích thước mới
    assert(A.size() == rows * cols && "Số phần tử không khớp với kích thước mới");

    // Tạo một ma trận tạm thời ánh xạ lại các phần tử của ma trận gốc
    MatrixXd B = Map<const MatrixXd>(A.data(), rows, cols);

    return B;
}

void train_model(const MatrixXd &x1,const MatrixXd &x2,const MatrixXd &u,int n_learn,const MatrixXd &R,const MatrixXd &Q1,const MatrixXd &Q2,const MatrixXd &Ks_12, MatrixXd &K12,vector<double> &dK) {
    int n = B_k(0).rows();
    int m = B_k(0).cols();
    for(int i=0;i<n_learn;i++) {
        MatrixXd Z = MatrixXd::Zero(f-1, n*(n+1)/2 + p*m*n + p*m*(p*m+1)/2);
        MatrixXd Y = MatrixXd::Zero(f-1, 1);
        for (int k = 0 ;k<f-1;k++) {
            Z.block(k,0,1,n*(n+1)/2) = (vecv(x2.block(0,k,n,1))-vecv(x2.block(0,k+1,n,1))).transpose();
            Z.block(k,n*(n+1)/2,1,p*m*n) = 2*kroneckerProduct((K12*x2.block(0,k+1,n,1)).transpose(),(x2.block(0,k+1,n,1)).transpose())
                           + 2*kroneckerProduct((u.block(0,k,p*m,1)).transpose(),(x2.block(0,k,n,1)).transpose());
            Z.block(k,n*(n+1)/2+p*m*n,1,p*m*(p*m+1)/2)=(vecv(u.block(0,k,p*m,1))-vecv(K12*x2.block(0,k+1,n,1))).transpose();
            Y.block(k,0,1,1) =(u.block(0,k,p*m,1)).transpose()*R*u.block(0,k,p*m,1)
             + (x1.block(0,k+1,(p-1)*n,1)).transpose()*Q1*x1.block(0,k+1,(p-1)*n,1)
             + (x2.block(0,k,n,1)).transpose()*Q2*x2.block(0,k,n,1);
        }
        MatrixXd vec_H(n*(n+1)/2 + p*m*n + p*m*(p*m+1)/2,1);
        vec_H = (Z.transpose()*Z).inverse()*Z.transpose()*Y;
        MatrixXd theta(n*(n+1)/2,1);
        theta = vec_H.block(0,0,n*(n+1)/2,1);
        MatrixXd H11(n,n);
        H11 = vecs_inv(theta);
        MatrixXd H12(n,p*m);
        H12 = reshape(vec_H.block(n*(n+1)/2,0,n*p*m,1),n,p*m);
        MatrixXd theta1(p*m*(p*m+1)/2,1);
        theta1 = vec_H.block(n*(n+1)/2+p*m*n,0,p*m*(p*m+1)/2,1);
        MatrixXd H22 = vecs_inv(theta1);
        K12 = H22.inverse()*H12.transpose();
        MatrixXd dKs(p*m,n);
        dKs = (K12-Ks_12);
        double dK_s = dKs.norm();
        dK.push_back(dK_s);
    }
}

void saveVectorToFile(const std::vector<double>& vec, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (const auto& value : vec) {
            file << value << "\n";  // Ghi từng giá trị trên một dòng
        }
        file.close();
        std::cout << "Vector saved to " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
    }
}

int main() {
    int n = B_k(0).rows();
    int m = B_k(0).cols();
    MatrixXd A(p*n,p*n);
    MatrixXd B(p*n,p*m);
    MatrixXd A1((p-1)*n,n);
    MatrixXd A2(n,n);
    MatrixXd B1((p-1)*n,p*m);
    MatrixXd B2(n,p*m);
    MatrixXd Q(p*n,p*n);
    MatrixXd R(p*m,p*m);
    MatrixXd Q1((p-1)*n,(p-1)*n);
    MatrixXd Q2(n,n);
    MatrixXd R1((p-1)*m,(p-1)*m);
    MatrixXd R2(m,m);
    MatrixXd Q0 = MatrixXd::Identity(n, n);
    MatrixXd R0 = MatrixXd::Identity(m, m);
    find_AB(A,B,A1,A2,B1,B2);
    find_QR(Q,R,Q1,Q2,R1,R2,Q0,R0);
    MatrixXd K0_12(p*m,n);
    K0_12 << 0.3650,0.5898,
             0.2102,0.4093,
             0.0226,0.1207,
             0.0352,-0.0860,
             0.1895,-0.1610,
             0.2523,-0.1609,
             0.1427,-0.1229,
             0.0158,-0.0479,
             0.0214,0.0488,
             0.1165,0.1235;
    MatrixXd K0(p*m,p*n);
    K0 <<  MatrixXd::Zero(p * m, (p - 1) * n),K0_12;
    MatrixXd x0_0(n,1); x0_0 << 3,-2;
    MatrixXd x(p*n,f+1);
    MatrixXd x1((p-1)*n,f+1);
    MatrixXd x2(n,f+1);
    MatrixXd u(p*m,f);
    x.block(0,0,p*n,1) << MatrixXd::Zero((p-1) * n, 1),x0_0;
    x1.block(0,0,(p-1)*n,1) = x.block(0,0,(p-1)*n,1);
    x2.block(0,0,n,1) = x.block((p-1)*n,0,n,1);
    colect_data(x,x1,x2,u,A,B,K0);
    int n_learn = 20;
    MatrixXd K12(p*m,p*n);
    K12 = K0_12;
    // MatrixXd dK(n_learn,1);
    vector<double> dK;
    MatrixXd Ks_12(p*m,n);
    Ks_12 << 0.7035  ,  0.7127,
             0.4002  ,  0.4866,
             0.0322  ,  0.1329,
             0.0962  , -0.0786,
             0.4853  , -0.1056,
             0.6575  , -0.0858,
             0.3806  , -0.0921,
             0.0382   ,-0.0574,
             0.0757  ,  0.0822,
             0.4053   , 0.2594;
    train_model(x1,x2,u,n_learn,R,Q1,Q2,Ks_12,K12,dK);
    saveVectorToFile(dK, "output.txt");
    cout << "Result: " <<endl << dK[15] << endl;
    return 0;
}






