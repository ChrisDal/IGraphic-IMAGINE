// -------------------------------------------
// gMini : a minimal OpenGL/GLUT application
// for 3D graphics.
// Copyright (C) 2006-2008 Tamy Boubekeur
// All rights reserved.
// -------------------------------------------

// -------------------------------------------
// Disclaimer: this code is dirty in the
// meaning that there is no attention paid to
// proper class attribute access, memory
// management or optimisation of any kind. It
// is designed for quick-and-dirty testing
// purpose.
// -------------------------------------------

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <GL/glut.h>
#include <float.h>
#include "src/Vec3.h"
#include "src/Camera.h"



// Out of Core simplification 
// https://web.cse.ohio-state.edu/~shen.94/Su01_888/lindstro.pdf

struct OCS {

    float m_espilon = (float)1.0e-2;
    float m_quadric[16]; 
    float n[4];
    Mat3 m_A; // Q = (A , -b ; -bT , c)
    Vec3 m_b; 

    inline OCS()
    {
        n[0] = 0.0f;  n[1] = 0.0f; n[2] = 0.0f;  n[3] = 0.0f; 
        for (unsigned int k = 0; k < 16; k++)
        {
            m_quadric[k] = 0.0f; 
        }
        m_A = Mat3();
        m_b = Vec3(0.0f, 0.0f, 0.0f); 
    }; 

    void reset() {
        n[0] = 0.0f;  n[1] = 0.0f; n[2] = 0.0f;  n[3] = 0.0f;
        for (unsigned int k = 0; k < 16; k++)
        {
            m_quadric[k] = 0.0f;
        }

        m_A = Mat3(); 

        m_b[0] = 0.0f; 
        m_b[1] = 0.0f; 
        m_b[2] = 0.0f; 
    }

    // Solve Ax = b
    Vec3 solveA(const Vec3& cellcenter)
    {
        setA(); 
        setB(); 
        
        // SVD Decomposition 
        Mat3 U, Vt; 
        float sx, sy, sz; // Sigma
        m_A.SVD(U, sx, sy, sz, Vt); 
        Vec3 sxyz = Vec3(sx, sy, sz); 
        //std::cout << "Singular Values : " << sxyz << std::endl; 
        Mat3 sigmapinv = processPseudoInvSigma(sxyz, m_espilon);
        // our vertice approximation
        Vec3 x; 
        // x = x^ + V * S+ * U.T * ( b - A * x^)
        x = cellcenter + Vt.getTranspose() * sigmapinv * U.getTranspose() * (m_b - m_A * cellcenter); 

        return x; 
    }

    // discard lowest singular values and inverse matrix => stability 
    Mat3 processPseudoInvSigma(const Vec3& sxyz, float epsilon)
    {
        Mat3 sigma = Mat3::Identity(); 
        // SVD with ordered singular values from gsl 
        // largest singular value = s[0]
        sigma(0, 0) = 1.0f / sxyz[0];
        sigma(1, 1) = sxyz[1] / sxyz[0] > epsilon ? 1.0f / sxyz[1] : 0.0f;
        sigma(2, 2) = sxyz[2] / sxyz[0] > epsilon ? 1.0f / sxyz[2] : 0.0f;

        return sigma; 
        
    }

    inline void setA()
    {
        m_A = Mat3();
        m_A(0, 0) = m_quadric[0];
        m_A(0, 1) = m_quadric[1];
        m_A(0, 2) = m_quadric[2];
    
        m_A(1, 0) = m_quadric[4];
        m_A(1, 1) = m_quadric[5];
        m_A(1, 2) = m_quadric[6];
    
        m_A(2, 0) = m_quadric[8];
        m_A(2, 1) = m_quadric[9];
        m_A(2, 2) = m_quadric[10];
    }

    inline void setB() {
        m_b = - 1.0f * Vec3(m_quadric[12], m_quadric[13], m_quadric[14]) ;
    }
    // for each triangle we add the quadric 
    void addingUpQ(const Vec3& x1, const Vec3& x2, const Vec3& x3)
    {
        // Process N
        Vec3 wnorm = Vec3::cross(x1, x2) + Vec3::cross(x2, x3) + Vec3::cross(x3, x1); 
        n[3] = - scalar3Product(x1, x2, x3); 
        n[0] = wnorm[0]; 
        n[1] = wnorm[1];
        n[2] = wnorm[2];


        // Q = n nT : row major
        // 1st row 
        m_quadric[0] += n[0] * n[0]; 
        m_quadric[1] += n[0] * n[1]; 
        m_quadric[2] += n[0] * n[2]; 
        m_quadric[3] += n[0] * n[3]; 
        // 2nd row   
        m_quadric[4] += n[1] * n[0];
        m_quadric[5] += n[1] * n[1];
        m_quadric[6] += n[1] * n[2];
        m_quadric[7] += n[1] * n[3];
                     
        // 3rd row   
        m_quadric[8] += n[2] * n[0];
        m_quadric[9] += n[2] * n[1];
        m_quadric[10]+= n[2] * n[2];
        m_quadric[11]+= n[2] * n[3];

        // 4th row => we can discard the last line if needed 
        m_quadric[12] += n[3] * n[0];
        m_quadric[13] += n[3] * n[1];
        m_quadric[14] += n[3] * n[2];
        m_quadric[15] += n[3] * n[3];
    }

    inline float scalar3Product(const Vec3& a, const Vec3& b, const Vec3& c)
    {
        return Vec3::dot(Vec3::cross(a, b), c); 
    }


    

};

// VecDictPy is a python-like dictionnary specialized in Vec3f-key
struct VecDictPy {

    std::vector<std::vector<int>> index; 
    std::vector<Vec3> points; 
    std::vector<OCS> quadrics; 

    float epsilon = (float)1.0e-7; 

    inline VecDictPy() {}; 
    inline ~VecDictPy() {}; 
    // handling existing or not point 
    void add(const Vec3& point, int idx) {
        int ind = findIndex(point); 
        
        if (ind == -1)
        {
            points.push_back(point); 
            std::vector<int> tmpvec = { idx }; 
            index.push_back(tmpvec); 
        }
        else
        {
            index[ind].push_back(idx);
        }
    }

    // more accurate for float 
    inline bool equals(const Vec3& a, const Vec3& b) {
        return std::abs(a[0] - b[0]) < epsilon && std::abs(a[1] - b[1]) < epsilon && std::abs(a[2] - b[2]) < epsilon;
    }

    bool contains(const Vec3& point, bool useEqual = false) {

        if (!useEqual) {
            return findIndex(point) != -1; 
        }
        else
        {
            return findIndexEqual(point) != -1;
        }
        
    }

    // retrieve by index 
    std::vector<int> operator [] (unsigned int c) const { return index[c];  }
    std::vector<int>& operator [] (unsigned int c) { return index[c]; };
   
    // Find index : use operator == from Vec3 
    
    int findIndex(const Vec3& point)
    {
        for (size_t pk = 0; pk < points.size(); pk++) {
            if (points[pk] == point) {
                return pk;
            }
        }

        return -1; 
    }

    // use VecDictPy equals function
    int findIndexEqual(const Vec3& point)
    {
        for (size_t pk = 0; pk < points.size(); pk++) {
            if (equals(points[pk], point)) {
                return pk;
            }
        }

        return -1;
    }

    int processNindex() {
        int nindex = 0; 
        for (size_t kni = 0; kni < index.size(); kni++)
        {
            nindex += index[kni].size(); 
        }

        return nindex; 
    }

    // old pythonic-way to retrieve keys and values (copy)
    std::vector<Vec3> keys() { return points; }
    std::vector<std::vector<int>> values() { return index; }

    void printInfos() {
        std::cout << "Npoints = " << points.size() << std::endl;
        std::cout << "Number of indices = " << processNindex()  << std::endl;
    }

    void initQuadrics()
    {
        quadrics = std::vector<OCS>(points.size()); 
    }
};

struct Triangle {
    inline Triangle () {
        v[0] = v[1] = v[2] = 0;
    }
    inline Triangle (const Triangle & t) {
        v[0] = t.v[0];   v[1] = t.v[1];   v[2] = t.v[2];
    }
    inline Triangle (unsigned int v0, unsigned int v1, unsigned int v2) {
        v[0] = v0;   v[1] = v1;   v[2] = v2;
    }
    unsigned int & operator [] (unsigned int iv) { return v[iv]; }
    unsigned int operator [] (unsigned int iv) const { return v[iv]; }
    inline virtual ~Triangle () {}
    inline Triangle & operator = (const Triangle & t) {
        v[0] = t.v[0];   v[1] = t.v[1];   v[2] = t.v[2];
        return (*this);
    }

    inline bool operator==(const Triangle& rhs) const {
        return v[0] == rhs[0] && v[1] == rhs[1] && v[2] == rhs[2]; 
    }

    inline bool operator!=(const Triangle& rhs) const {
        return !(*this == rhs);
    }

    // membres :
    unsigned int v[3];
};

// ------------------------------------
// File I/O
// ------------------------------------
bool saveOFF(const std::string& filename,
    std::vector< Vec3 >& i_vertices,
    std::vector< Vec3 >& i_normals,
    std::vector< Triangle >& i_triangles,
    bool save_normals = true) {
    std::ofstream myfile;
    myfile.open(filename.c_str());
    if (!myfile.is_open()) {
        std::cout << filename << " cannot be opened" << std::endl;
        return false;
    }

    myfile << "OFF" << std::endl;

    unsigned int n_vertices = i_vertices.size(), n_triangles = i_triangles.size();
    myfile << n_vertices << " " << n_triangles << " 0" << std::endl;

    for (unsigned int v = 0; v < n_vertices; ++v) {
        myfile << i_vertices[v][0] << " " << i_vertices[v][1] << " " << i_vertices[v][2] << " ";
        if (save_normals) myfile << i_normals[v][0] << " " << i_normals[v][1] << " " << i_normals[v][2] << std::endl;
        else myfile << std::endl;
    }
    for (unsigned int f = 0; f < n_triangles; ++f) {
        myfile << 3 << " " << i_triangles[f][0] << " " << i_triangles[f][1] << " " << i_triangles[f][2];
        myfile << std::endl;
    }
    myfile.close();
    return true;
}

void openOFF(std::string const& filename,
    std::vector<Vec3>& o_vertices,
    std::vector<Vec3>& o_normals,
    std::vector< Triangle >& o_triangles)
{
    std::ifstream myfile;
    myfile.open(filename.c_str());
    if (!myfile.is_open())
    {
        std::cout << filename << " cannot be opened" << std::endl;
        return;
    }

    std::string magic_s;

    myfile >> magic_s;

    if (magic_s != "OFF")
    {
        std::cout << magic_s << " != OFF :   We handle ONLY *.off files." << std::endl;
        myfile.close();
        exit(1);
    }

    int n_vertices, n_faces, dummy_int;
    myfile >> n_vertices >> n_faces >> dummy_int;

    o_vertices.clear();

    for (int v = 0; v < n_vertices; ++v)
    {
        float x, y, z;

        myfile >> x >> y >> z;
        o_vertices.push_back(Vec3(x, y, z));

    }

    o_triangles.clear();
    for (int f = 0; f < n_faces; ++f)
    {
        int n_vertices_on_face;
        myfile >> n_vertices_on_face;

        if (n_vertices_on_face == 3)
        {
            unsigned int _v1, _v2, _v3;
            myfile >> _v1 >> _v2 >> _v3;

            o_triangles.push_back(Triangle(_v1, _v2, _v3));
        }
        else if (n_vertices_on_face == 4)
        {
            unsigned int _v1, _v2, _v3, _v4;
            myfile >> _v1 >> _v2 >> _v3 >> _v4;

            o_triangles.push_back(Triangle(_v1, _v2, _v3));
            o_triangles.push_back(Triangle(_v1, _v3, _v4));
        }
        else
        {
            std::cout << "We handle ONLY *.off files with 3 or 4 vertices per face" << std::endl;
            myfile.close();
            exit(1);
        }
    }

}

// ------------------------------------

Vec3 getRepresentant(const std::vector<std::vector<std::vector<Vec3>>>& g_grid, const Vec3& v, const Vec3& dxyz, const Vec3& bbmin)
{
    int xi = int((v[0]-bbmin[0]) / dxyz[0]);
    int yi = int((v[1]-bbmin[1]) / dxyz[1]);
    int zi = int((v[2]-bbmin[2]) / dxyz[2]);

    Vec3 center = g_grid[xi][yi][zi] +  Vec3(dxyz / 2.0f);
    return center;
}

Triangle getCoordRepr(const std::vector<std::vector<std::vector<Vec3>>>& g_grid, const Vec3& v, const Vec3& dxyz, const Vec3& bbmin) 
{
    int xi = int((v[0] - bbmin[0]) / dxyz[0]);
    int yi = int((v[1] - bbmin[1]) / dxyz[1]);
    int zi = int((v[2] - bbmin[2]) / dxyz[2]);

    if (xi == 0 && yi == 0 && zi == 0) {
        std::cout << "Coord = 0 0 0 for Vec " << v << std::endl; 
    }

    if (g_grid[xi][yi][zi].length() < 0.00005f) {
        std::cout << "Vec center ==  " << g_grid[xi][yi][zi] << std::endl;
    }

    return Triangle(xi, yi, zi); 
}


struct Mesh {
    std::vector< Vec3 > vertices; // List of mesh vertices positions
    std::vector< Vec3 > normals;
    std::vector< Triangle > triangles;
    std::vector< Vec3 > triangle_normals;

    

    std::vector<std::vector<std::vector<Vec3>>> grid; 

    void computeTrianglesNormals(){

        triangle_normals.clear();
        for( unsigned int i = 0 ; i < triangles.size() ;i++ ){
            const Vec3 & e0 = vertices[triangles[i][1]] - vertices[triangles[i][0]];
            const Vec3 & e1 = vertices[triangles[i][2]] - vertices[triangles[i][0]];
            Vec3 n = Vec3::cross( e0, e1 );
            n.normalize();
            triangle_normals.push_back( n );
        }
    }

    void computeVerticesNormals(){

        normals.clear();
        normals.resize( vertices.size(), Vec3(0., 0., 0.) );
        for( unsigned int i = 0 ; i < triangles.size() ;i++ ){
            for( unsigned int t = 0 ; t < 3 ; t++ )
                normals[ triangles[i][t] ] += triangle_normals[i];
        }
        for( unsigned int i = 0 ; i < vertices.size() ;i++ )
            normals[ i ].normalize();

    }

    void computeNormals(){
        computeTrianglesNormals();
        computeVerticesNormals();
    }

    
    void printInfos()
    {
        std::cout << "Mesh Vertices = " << vertices.size() << std::endl; 
        std::cout << "Mesh Triangles = " << triangles.size() << std::endl; 
    }

    

    Vec3 computeGrid(const unsigned int& res, const Vec3& bbmin, const Vec3& bbmax) 
    {
        grid.clear();
        Vec3 dxyz = Vec3((bbmax[0] - bbmin[0]), (bbmax[1] - bbmin[1]), (bbmax[2] - bbmin[2]));
        dxyz /= static_cast<float>(res);

        std::vector < std::vector < Vec3 >> vecy;
        std::vector < Vec3 > vecz;
        for (unsigned int kx = 0; kx < res+1; kx++)
        {
            vecy.clear();
            vecy.reserve(res);
            for (unsigned int ky = 0; ky < res+1; ky++)
            {
                vecz.clear();
                vecz.reserve(res);
                for (unsigned int kz = 0; kz < res+1; kz++)
                {
                    vecz.emplace_back(bbmin[0] + (float)kx * dxyz[0], 
                                        bbmin[1] + (float)ky * dxyz[1], 
                                        bbmin[2] + (float)kz * dxyz[2]);
                }
                vecy.push_back(vecz);
            }

            grid.push_back(vecy);
        }

        return dxyz; 
    }

    void simplify(unsigned int resolution, const Vec3& bbmin, const Vec3& bbmax)
    {
        Vec3 dxyz = computeGrid(resolution, bbmin, bbmax);

        // Pour chaque sommet v du maillage, 
        // ajouter sa position et sa normale au sommet représentant de la cellule de G contenant v. 
        // Compter le nombre de sommets par cellule.

        VecDictPy representants2;
        for (size_t k = 0; k < vertices.size(); k++)
        {
            Vec3 center = getRepresentant(grid, vertices[k], dxyz, bbmin);
            representants2.add(center, k); 
        }

        std::cout << "-------------------------\n"; 
        representants2.printInfos(); 
        

        // Vertices 
        std::vector<Vec3> simpVertices = representants2.keys(); 
        unsigned int ntotalclassified = 0; 
        for (unsigned int k = 0; k < representants2.index.size(); k++)
        {
            ntotalclassified += representants2[k].size(); 
        }

        std::cout << "Number of vertices linked with grid : " << ntotalclassified << std::endl;
        std::cout << "-------------------------\n";

        // Triangles 
        std::vector<Triangle> simpTriangles; 
        int validTrig = 0; 
        for (size_t k = 0; k < triangles.size(); k++)
        {
            Vec3 v0 = getRepresentant(grid, vertices[triangles[k][0]], dxyz, bbmin);
            Vec3 v1 = getRepresentant(grid, vertices[triangles[k][1]], dxyz, bbmin);
            Vec3 v2 = getRepresentant(grid, vertices[triangles[k][2]], dxyz, bbmin);

            bool eqFound = representants2.equals(v0, v1); 
            eqFound |= representants2.equals(v0, v2);
            eqFound |= representants2.equals(v1, v2);

            if (eqFound) {
                continue; 
            }

            //triangles[k][ki] => index du representant 
            bool notFound = false; 
            Triangle t;

            for (unsigned int kii = 0; kii < 3; kii++)
            {
                int vIdx = representants2.findIndex(getRepresentant(grid, vertices[triangles[k][kii]], dxyz, bbmin)); 
                
                notFound |= vIdx == -1;
                if (notFound) { 
                    break; 
                }

                t[kii] = vIdx;
            }
            
            if (notFound) {
                continue; 
            }

            simpTriangles.push_back(t); 
            validTrig++; 

        }


        std::cout << "Valid Triangles = " << validTrig << std::endl; 
        triangles.clear(); 
        triangles.resize(simpTriangles.size());
        triangles = simpTriangles; 

        // Vertices and normals
        std::vector<Vec3> simpNorms(simpVertices.size(), Vec3(0.0f, 0.0f, 0.0f)); 

        for (size_t k = 0; k < simpVertices.size() ; k++)
        {
            Vec3 pos = Vec3(0.0f, 0.0f, 0.0f);
            // get index of vertices for points[k]
            for (const int& kr : representants2[k])
            {
                pos   += vertices[kr];
                simpNorms[k] += normals[kr];
            }

            pos /= (float)representants2[k].size();
            simpVertices[k] = pos;

            simpNorms[k].normalize();
            
        }

        vertices.clear(); 
        vertices = simpVertices; 
        normals.clear(); 
        normals = simpNorms;

        computeNormals(); 
    }

    void simplifyOCS(unsigned int resolution, const Vec3& bbmin, const Vec3& bbmax)
    {
        Vec3 dxyz = computeGrid(resolution, bbmin, bbmax);

        // Pour chaque sommet v du maillage, 
        // ajouter sa position et sa normale au sommet représentant de la cellule de G contenant v. 
        // Compter le nombre de sommets par cellule.

        VecDictPy representants;
        for (size_t k = 0; k < vertices.size(); k++)
        {
            Vec3 center = getRepresentant(grid, vertices[k], dxyz, bbmin);
            representants.add(center, k);
        }

        std::cout << "-------------------------\n";
        representants.printInfos();
        representants.initQuadrics(); 


        // Vertices 
        std::vector<Vec3> simpVertices = representants.keys();
        unsigned int ntotalclassified = 0;
        for (unsigned int k = 0; k < representants.index.size(); k++)
        {
            ntotalclassified += representants[k].size();
        }

        std::cout << "Number of vertices linked with grid : " << ntotalclassified << std::endl;
        std::cout << "-------------------------\n";

        // Triangles 
        std::vector<Triangle> simpTriangles;
        int validTrig = 0;

        for (size_t k = 0; k < triangles.size(); k++)
        {
            Vec3 v0 = getRepresentant(grid, vertices[triangles[k][0]], dxyz, bbmin);
            Vec3 v1 = getRepresentant(grid, vertices[triangles[k][1]], dxyz, bbmin);
            Vec3 v2 = getRepresentant(grid, vertices[triangles[k][2]], dxyz, bbmin);

            

            bool eqFound = representants.equals(v0, v1);
            eqFound |= representants.equals(v0, v2);
            eqFound |= representants.equals(v1, v2);

            if (eqFound) {
                continue;
            }

            //triangles[k][ki] => index du representant 
            bool notFound = false;
            Triangle t;

            for (unsigned int kii = 0; kii < 3; kii++)
            {
                int vIdx = representants.findIndex(getRepresentant(grid, vertices[triangles[k][kii]], dxyz, bbmin));

                notFound |= vIdx == -1;
                if (notFound) {
                    break;
                }

                t[kii] = vIdx;
            }

            if (notFound) {
                continue;
            }

            // Compute Associated Quadric 
            for (unsigned int kq = 0; kq < 3; kq++) {
                int indexVec = representants.findIndex(getRepresentant(grid, vertices[triangles[k][kq]], dxyz, bbmin));
                representants.quadrics[indexVec].addingUpQ(vertices[triangles[k][0]],
                                                            vertices[triangles[k][1]],
                                                            vertices[triangles[k][2]]);
            }

            simpTriangles.push_back(t);
            validTrig++;

        }


        std::cout << "Valid Triangles = " << validTrig << std::endl;
        triangles.clear();
        triangles.resize(simpTriangles.size());
        triangles = simpTriangles;

        // Vertices and normals
        std::vector<Vec3> simpNorms(simpVertices.size(), Vec3(0.0f, 0.0f, 0.0f));

        for (size_t k = 0; k < simpVertices.size(); k++)
        {
            Vec3 pos = Vec3(0.0f, 0.0f, 0.0f);
            // get index of vertices for points[k]
            for (const int& kr : representants[k])
            {
                simpNorms[k] += normals[kr];
            }

            // MSE 
            simpVertices[k] = representants.quadrics[k].solveA(representants.points[k]);

            simpNorms[k].normalize();

        }

        vertices.clear();
        vertices = simpVertices;
        normals.clear();
        normals = simpNorms;

        computeNormals();
    }

    void init(const std::string& filename) {

        openOFF(filename, vertices, normals, triangles);
        computeNormals();
    }

    void reset() {
        vertices.clear(); // List of mesh vertices positions
        normals.clear();
        triangles.clear();
        triangle_normals.clear();
    }

    void reset(const std::string& filename) {
        reset(); 
        init(filename); 
    }
};




// -------------------------------------------
// Global Variables 
// -------------------------------------------

Mesh mesh;
Vec3 bbmin;
Vec3 bbmax;
unsigned int resolution = 16; 
std::string filename = "data/avion_64.off"; 

bool display_normals;
bool display_smooth_normals;
bool display_mesh;
bool display_grid; 
bool display_vertices; 



// -------------------------------------------
// OpenGL/GLUT application code.
// -------------------------------------------

static GLint window;
static unsigned int SCREENWIDTH = 1600;
static unsigned int SCREENHEIGHT = 900;
static Camera camera;
static bool mouseRotatePressed = false;
static bool mouseMovePressed = false;
static bool mouseZoomPressed = false;
static int lastX=0, lastY=0, lastZoom=0;
static bool fullScreen = false;


// ------------------------------------




void computeBBox( std::vector< Vec3 > const & i_positions, Vec3 & bboxMin, Vec3 & bboxMax ) {
    bboxMin = Vec3 ( FLT_MAX , FLT_MAX , FLT_MAX );
    bboxMax = Vec3 ( FLT_MIN , FLT_MIN , FLT_MIN );
    for(unsigned int pIt = 0 ; pIt < i_positions.size() ; ++pIt) {
        for( unsigned int coord = 0 ; coord < 3 ; ++coord ) {
            bboxMin[coord] = std::min<float>( bboxMin[coord] , i_positions[pIt][coord] );
            bboxMax[coord] = std::max<float>( bboxMax[coord] , i_positions[pIt][coord] );
        }
    }
}

// ------------------------------------
// Application initialization
// ------------------------------------
void initLight () {
    GLfloat light_position1[4] = {22.0f, 16.0f, 50.0f, 0.0f};
    GLfloat direction1[3] = {-52.0f,-16.0f,-50.0f};
    GLfloat color1[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat ambient[4] = {0.3f, 0.3f, 0.3f, 0.5f};

    glLightfv (GL_LIGHT1, GL_POSITION, light_position1);
    glLightfv (GL_LIGHT1, GL_SPOT_DIRECTION, direction1);
    glLightfv (GL_LIGHT1, GL_DIFFUSE, color1);
    glLightfv (GL_LIGHT1, GL_SPECULAR, color1);
    glLightModelfv (GL_LIGHT_MODEL_AMBIENT, ambient);
    glEnable (GL_LIGHT1);
    glEnable (GL_LIGHTING);
}

void init () {
    camera.resize (SCREENWIDTH, SCREENHEIGHT);
    initLight ();
    glCullFace (GL_BACK);
    glDisable (GL_CULL_FACE);
    glDepthFunc (GL_LESS);
    glEnable (GL_DEPTH_TEST);
    glClearColor (0.2f, 0.2f, 0.3f, 1.0f);
    glEnable(GL_COLOR_MATERIAL);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

    display_normals = false;
    display_mesh = true;
    display_smooth_normals = true;
    display_grid = false; 
    display_vertices = false; 

}


// ------------------------------------
// Rendering.
// ------------------------------------

void drawVector( Vec3 const & i_from, Vec3 const & i_to ) {

    glBegin(GL_LINES);
    glVertex3f( i_from[0] , i_from[1] , i_from[2] );
    glVertex3f( i_to[0] , i_to[1] , i_to[2] );
    glEnd();
}

void drawAxis( Vec3 const & i_origin, Vec3 const & i_direction ) {

    glLineWidth(4); // for example...
    drawVector(i_origin, i_origin + i_direction);
}


void drawSmoothTriangleMesh( Mesh const & i_mesh ) {
    glBegin(GL_TRIANGLES);
    for(unsigned int tIt = 0 ; tIt < i_mesh.triangles.size(); ++tIt) {
        Vec3 p0 = i_mesh.vertices[i_mesh.triangles[tIt][0]];
        Vec3 n0 = i_mesh.normals[i_mesh.triangles[tIt][0]];

        Vec3 p1 = i_mesh.vertices[i_mesh.triangles[tIt][1]];
        Vec3 n1 = i_mesh.normals[i_mesh.triangles[tIt][1]];

        Vec3 p2 = i_mesh.vertices[i_mesh.triangles[tIt][2]];
        Vec3 n2 = i_mesh.normals[i_mesh.triangles[tIt][2]];

        glNormal3f( n0[0] , n0[1] , n0[2] );
        glVertex3f( p0[0] , p0[1] , p0[2] );
        glNormal3f( n1[0] , n1[1] , n1[2] );
        glVertex3f( p1[0] , p1[1] , p1[2] );
        glNormal3f( n2[0] , n2[1] , n2[2] );
        glVertex3f( p2[0] , p2[1] , p2[2] );
    }
    glEnd();

}

void drawTriangleMesh( Mesh const & i_mesh ) {
    glBegin(GL_TRIANGLES);
    for(unsigned int tIt = 0 ; tIt < i_mesh.triangles.size(); ++tIt) {
        Vec3 p0 = i_mesh.vertices[i_mesh.triangles[tIt][0]];
        Vec3 p1 = i_mesh.vertices[i_mesh.triangles[tIt][1]];
        Vec3 p2 = i_mesh.vertices[i_mesh.triangles[tIt][2]];

        Vec3 n = i_mesh.triangle_normals[tIt];

        glNormal3f( n[0] , n[1] , n[2] );

        glVertex3f( p0[0] , p0[1] , p0[2] );
        glVertex3f( p1[0] , p1[1] , p1[2] );
        glVertex3f( p2[0] , p2[1] , p2[2] );
    }
    glEnd();

}

void drawMesh( Mesh const & i_mesh ){
    if(display_smooth_normals)
        drawSmoothTriangleMesh(i_mesh) ;
    else {
        drawTriangleMesh(i_mesh) ;
    }
}

void drawVectorField( std::vector<Vec3> const & i_positions, std::vector<Vec3> const & i_directions ) {
    glLineWidth(1.);
    for(unsigned int pIt = 0 ; pIt < i_directions.size() ; ++pIt) {
        Vec3 to = i_positions[pIt] + 0.02*i_directions[pIt];
        drawVector(i_positions[pIt], to);
    }
}

void drawNormals(Mesh const& i_mesh){

    if(display_smooth_normals){
        drawVectorField( i_mesh.vertices, i_mesh.normals );
    } else {
        std::vector<Vec3> triangle_baricenters;
        for ( const Triangle& triangle : i_mesh.triangles ){
            Vec3 triangle_baricenter (0.,0.,0.);
            for( unsigned int i = 0 ; i < 3 ; i++ )
                triangle_baricenter += i_mesh.vertices[triangle[i]];
            triangle_baricenter /= 3;
            triangle_baricenters.push_back(triangle_baricenter);
        }

        drawVectorField( triangle_baricenters, i_mesh.triangle_normals );
    }
}

void drawPointSet(std::vector< Vec3 > const& i_positions, std::vector< Vec3 > const& i_normals) {
    glBegin(GL_POINTS);
    for (unsigned int pIt = 0; pIt < i_positions.size(); ++pIt) {
        glNormal3f(i_normals[pIt][0], i_normals[pIt][1], i_normals[pIt][2]);
        glVertex3f(i_positions[pIt][0], i_positions[pIt][1], i_positions[pIt][2]);
    }
    glEnd();
}

// Wrapper color and size for pointset
void drawPoints(std::vector<Vec3>const& projpos, std::vector<Vec3>const& projnorm,
    float nR = 1.0f, float nG = 0.5f, float nB = 0.5f, int pointsize = 4)
{
    glPointSize(pointsize);
    glColor3f(nR, nG, nB);
    drawPointSet(projpos, projnorm);
}


// Wrapper color and size for grid 
void drawGrid(std::vector < std::vector < std::vector<Vec3>>>const& gridset, float nR = 1.0f, float nG = 0.5f,
                float nB = 0.5f, int pointsize = 4)
{
    glPointSize(pointsize);
    glColor3f(nR, nG, nB);

    Vec3 sameNormal = Vec3(1.0f, 0.0, 0.0); 
    Vec3 pointsetSize = Vec3(gridset.size(), gridset[0].size(), gridset[0][0].size());

    glBegin(GL_POINTS);
    for (unsigned int pItx = 0; pItx < (unsigned int)pointsetSize[0]; ++pItx)
    {
        for (unsigned int pIty = 0; pIty < (unsigned int)pointsetSize[1]; ++pIty)
        {
            for (unsigned int pItz = 0; pItz < (unsigned int)pointsetSize[2]; ++pItz)
            {
                glNormal3f(sameNormal[0], sameNormal[1], sameNormal[2]);
                glVertex3f(gridset[pItx][pIty][pItz][0],
                    gridset[pItx][pIty][pItz][1],
                    gridset[pItx][pIty][pItz][2]);

            }
        }
    }
    glEnd();
} 


//Draw fonction
void draw () {

    glColor3f(0.8,1,0.8);
    if (display_mesh)
    {
        drawMesh(mesh);
    }
    

    if(display_normals){
        glColor3f(1.,0.,0.);
        drawNormals(mesh);
    }

    if (display_grid) {
        drawGrid(mesh.grid);
    }

    if (display_vertices) {
        std::vector<Vec3> normnormals(mesh.vertices.size(), Vec3(1.0f, 0.0, 0.0)); 
        drawPoints(mesh.vertices, normnormals, 1.0f, 0.0, 1.0f, 8);
    }
}

void display () {
    
    glLoadIdentity ();
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    camera.apply ();
    draw ();
    glFlush ();
    glutSwapBuffers(); 
}

void idle () {
    glutPostRedisplay ();
}





// ------------------------------------
// User inputs
// ------------------------------------
//Keyboard event
void key (unsigned char keyPressed, int x, int y) {
    switch (keyPressed) {
    case 'f':
        if (fullScreen == true) {
            glutReshapeWindow (SCREENWIDTH, SCREENHEIGHT);
            fullScreen = false;
        } else {
            glutFullScreen ();
            fullScreen = true;
        }
        break;


    case 'w':
        GLint polygonMode[2];
        glGetIntegerv(GL_POLYGON_MODE, polygonMode);
        if(polygonMode[0] != GL_FILL)
            glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        else
            glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
        break;

    case 'm': //Display Mesh
        display_mesh = !display_mesh;
        break;

    case 'n': //Press n key to display normals
        display_normals = !display_normals;
        break;


    case 's': //Switches between face normals and vertices normals
        display_smooth_normals = !display_smooth_normals;
        break;

    case 'g': 
        display_grid = !display_grid; 
        break; 

    case 'y': // Simplify 
        std::cout << "Resolution = " << resolution << std::endl; 
        std::cout << "Mean Vertex Simplification" << std::endl;
        mesh.simplify(resolution, bbmin, bbmax); 
        break; 

    case 'u': // Simplify 
        std::cout << "Resolution = " << resolution << std::endl;
        std::cout << "Out-of-Core Simplification"  << std::endl;
        mesh.simplifyOCS(resolution, bbmin, bbmax);
        break;

    case 'r': // Reset Mesh
        mesh.reset(filename); 
        break;

    case 'a' : // Add a level in simplification 
        resolution *= 2; 
        std::cout << "Increase Resolution to " << resolution << "." << std::endl; 
        break; 

    case 'd' : // decrease resolution 
        resolution = resolution > 4  ? resolution / 2 : 4;
        std::cout << "Decrease Resolution to " << resolution << "." << std::endl;
        break;
    case 'p': // draw pointset 
        display_vertices = !display_vertices; 
        break; 

    default:
        break;
    }
    idle ();
}

//Mouse events
void mouse (int button, int state, int x, int y) {
    if (state == GLUT_UP) {
        mouseMovePressed = false;
        mouseRotatePressed = false;
        mouseZoomPressed = false;
    } else {
        if (button == GLUT_LEFT_BUTTON) {
            camera.beginRotate (x, y);
            mouseMovePressed = false;
            mouseRotatePressed = true;
            mouseZoomPressed = false;
        } else if (button == GLUT_RIGHT_BUTTON) {
            lastX = x;
            lastY = y;
            mouseMovePressed = true;
            mouseRotatePressed = false;
            mouseZoomPressed = false;
        } else if (button == GLUT_MIDDLE_BUTTON) {
            if (mouseZoomPressed == false) {
                lastZoom = y;
                mouseMovePressed = false;
                mouseRotatePressed = false;
                mouseZoomPressed = true;
            }
        }
    }
    idle ();
}

//Mouse motion, update camera
void motion (int x, int y) {
    if (mouseRotatePressed == true) {
        camera.rotate (x, y);
    }
    else if (mouseMovePressed == true) {
        camera.move ((x-lastX)/static_cast<float>(SCREENWIDTH), (lastY-y)/static_cast<float>(SCREENHEIGHT), 0.0);
        lastX = x;
        lastY = y;
    }
    else if (mouseZoomPressed == true) {
        camera.zoom (float (y-lastZoom)/SCREENHEIGHT);
        lastZoom = y;
    }
}


void reshape(int w, int h) {
    camera.resize (w, h);
}

// ------------------------------------
// Start of graphical application
// ------------------------------------
int main (int argc, char ** argv) {
    if (argc > 2) {
        exit (EXIT_FAILURE);
    }
    glutInit (&argc, argv);
    glutInitDisplayMode (GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize (SCREENWIDTH, SCREENHEIGHT);
    window = glutCreateWindow ("TP HAI714I");



    init ();
    glutIdleFunc (idle);
    glutDisplayFunc (display);
    glutKeyboardFunc (key);
    glutReshapeFunc (reshape);
    glutMotionFunc (motion);
    glutMouseFunc (mouse);
    key ('?', 0, 0);

    // Unit sphere mesh loaded with precomputed normals
    openOFF(filename, mesh.vertices, mesh.normals, mesh.triangles);
    mesh.computeNormals();

    // Compute bounding box 
    computeBBox(mesh.vertices, bbmin, bbmax); 
    double diagBBox = (bbmax - bbmin).length();
    Vec3 dxyz = Vec3(0.1f * diagBBox, 0.1f * diagBBox, 0.1f * diagBBox);
    bbmin -= dxyz; 
    bbmax += dxyz; 
    // Compute Grid at least once to display it 
    mesh.computeGrid(resolution, bbmin, bbmax); 

    glutMainLoop ();
    return EXIT_SUCCESS;
}

