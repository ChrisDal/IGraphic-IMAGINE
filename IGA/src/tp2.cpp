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
#include <cmath>
#include <string>
#include <cstdio>
#include <cstdlib>

#include <numeric>
#include <GL/glut.h>
#include <float.h>
#include "src/Vec3.h"
#include "src/Camera.h"
#include "src/jmkdtree.h"
#include "src/Voxel.h"


enum IMPL_VALUE { INTERNE = -1, ON_SURFACE = 0, EXTERNE = 1 };

std::vector< Vec3 > positions;
std::vector< Vec3 > normals;

std::vector< Vec3 > positions2;
std::vector< Vec3 > normals2;

std::vector< Vec3 > positions3;
std::vector< Vec3 > normals3;

std::vector< Vec3 > positions4;
std::vector< Vec3 > normals4;

std::vector< Vec3 > positions5;
std::vector< Vec3 > normals5;

// TP2 

std::vector<Vec3> new_grid;
std::vector<Vec3> grid_norms;

std::vector<Vec3> voxelSommets;
std::vector<Vec3> voxelSommetNorms;

std::vector<Vec3> surfTrig;

int Voxel::id = 0;

// -------------------------------------------
// OpenGL/GLUT application code.
// -------------------------------------------

static GLint window;
static unsigned int SCREENWIDTH = 640;
static unsigned int SCREENHEIGHT = 480;
static Camera camera;
static bool mouseRotatePressed = false;
static bool mouseMovePressed = false;
static bool mouseZoomPressed = false;
static int lastX = 0, lastY = 0, lastZoom = 0;
static bool fullScreen = false;
static bool displayModel = true;
static bool displayVoxelCenter = true;
static bool displaySurface = true;




// ------------------------------------------------------------------------------------------------------------
// i/o and some stuff
// ------------------------------------------------------------------------------------------------------------
void loadPN(const std::string& filename, std::vector< Vec3 >& o_positions, std::vector< Vec3 >& o_normals) {
    unsigned int surfelSize = 6;
    FILE* in;
    fopen_s(&in, filename.c_str(), "rb");
    if (in == NULL) {
        std::cout << filename << " is not a valid PN file." << std::endl;
        return;
    }
    size_t READ_BUFFER_SIZE = 1000; // for example...
    float* pn = new float[surfelSize * READ_BUFFER_SIZE];
    o_positions.clear();
    o_normals.clear();
    while (!feof(in)) {
        unsigned numOfPoints = fread(pn, 4, surfelSize * READ_BUFFER_SIZE, in);
        for (unsigned int i = 0; i < numOfPoints; i += surfelSize) {
            o_positions.push_back(Vec3(pn[i], pn[i + 1], pn[i + 2]));
            o_normals.push_back(Vec3(pn[i + 3], pn[i + 4], pn[i + 5]));
        }

        if (numOfPoints < surfelSize * READ_BUFFER_SIZE) break;
    }
    fclose(in);
    delete[] pn;
}
void savePN(const std::string& filename, std::vector< Vec3 > const& o_positions, std::vector< Vec3 > const& o_normals) {
    if (o_positions.size() != o_normals.size()) {
        std::cout << "The pointset you are trying to save does not contain the same number of points and normals." << std::endl;
        return;
    }
    FILE* outfile;
    fopen_s(&outfile, filename.c_str(), "wb");
    if (outfile == NULL) {
        std::cout << filename << " is not a valid PN file." << std::endl;
        return;
    }
    for (unsigned int pIt = 0; pIt < o_positions.size(); ++pIt) {
        fwrite(&(o_positions[pIt]), sizeof(float), 3, outfile);
        fwrite(&(o_normals[pIt]), sizeof(float), 3, outfile);
    }
    fclose(outfile);
}
void scaleAndCenter(std::vector< Vec3 >& io_positions) {
    Vec3 bboxMin(FLT_MAX, FLT_MAX, FLT_MAX);
    Vec3 bboxMax(FLT_MIN, FLT_MIN, FLT_MIN);
    for (unsigned int pIt = 0; pIt < io_positions.size(); ++pIt) {
        for (unsigned int coord = 0; coord < 3; ++coord) {
            bboxMin[coord] = std::min<float>(bboxMin[coord], io_positions[pIt][coord]);
            bboxMax[coord] = std::max<float>(bboxMax[coord], io_positions[pIt][coord]);
        }
    }
    Vec3 bboxCenter = (bboxMin + bboxMax) / 2.f;
    float bboxLongestAxis = std::max<float>(bboxMax[0] - bboxMin[0], std::max<float>(bboxMax[1] - bboxMin[1], bboxMax[2] - bboxMin[2]));
    for (unsigned int pIt = 0; pIt < io_positions.size(); ++pIt) {
        io_positions[pIt] = (io_positions[pIt] - bboxCenter) / bboxLongestAxis;
    }
}

void applyRandomRigidTransformation(std::vector< Vec3 >& io_positions, std::vector< Vec3 >& io_normals) {
    srand(time(NULL));
    Mat3 R = Mat3::RandRotation();
    Vec3 t = Vec3::Rand(1.f);
    for (unsigned int pIt = 0; pIt < io_positions.size(); ++pIt) {
        io_positions[pIt] = R * io_positions[pIt] + t;
        io_normals[pIt] = R * io_normals[pIt];
    }
}

void subsample(std::vector< Vec3 >& i_positions, std::vector< Vec3 >& i_normals, float minimumAmount = 0.1f, float maximumAmount = 0.2f) {
    std::vector< Vec3 > newPos, newNormals;
    std::vector< unsigned int > indices(i_positions.size());
    for (unsigned int i = 0; i < indices.size(); ++i) indices[i] = i;
    srand(time(NULL));
    std::random_shuffle(indices.begin(), indices.end());
    unsigned int newSize = indices.size() * (minimumAmount + (maximumAmount - minimumAmount) * (float)(rand()) / (float)(RAND_MAX));
    newPos.resize(newSize);
    newNormals.resize(newSize);
    for (unsigned int i = 0; i < newPos.size(); ++i) {
        newPos[i] = i_positions[indices[i]];
        newNormals[i] = i_normals[indices[i]];
    }
    i_positions = newPos;
    i_normals = newNormals;
}

bool save(const std::string& filename, std::vector< Vec3 >& vertices, std::vector< unsigned int >& triangles) {
    std::ofstream myfile;
    myfile.open(filename.c_str());
    if (!myfile.is_open()) {
        std::cout << filename << " cannot be opened" << std::endl;
        return false;
    }

    myfile << "OFF" << std::endl;

    unsigned int n_vertices = vertices.size(), n_triangles = triangles.size() / 3;
    myfile << n_vertices << " " << n_triangles << " 0" << std::endl;

    for (unsigned int v = 0; v < n_vertices; ++v) {
        myfile << vertices[v][0] << " " << vertices[v][1] << " " << vertices[v][2] << std::endl;
    }
    for (unsigned int f = 0; f < n_triangles; ++f) {
        myfile << 3 << " " << triangles[3 * f] << " " << triangles[3 * f + 1] << " " << triangles[3 * f + 2];
        myfile << std::endl;
    }
    myfile.close();
    return true;
}



// ------------------------------------------------------------------------------------------------------------
// rendering.
// ------------------------------------------------------------------------------------------------------------

void initLight() {
    GLfloat light_position1[4] = { 22.0f, 16.0f, 50.0f, 0.0f };
    GLfloat direction1[3] = { -52.0f,-16.0f,-50.0f };
    GLfloat color1[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat ambient[4] = { 0.3f, 0.3f, 0.3f, 0.5f };

    glLightfv(GL_LIGHT1, GL_POSITION, light_position1);
    glLightfv(GL_LIGHT1, GL_SPOT_DIRECTION, direction1);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, color1);
    glLightfv(GL_LIGHT1, GL_SPECULAR, color1);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);
    glEnable(GL_LIGHT1);
    glEnable(GL_LIGHTING);
}

void init() {
    camera.resize(SCREENWIDTH, SCREENHEIGHT);
    initLight();
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.2f, 0.2f, 0.3f, 1.0f);
    glEnable(GL_COLOR_MATERIAL);
}



void drawTriangleMesh(std::vector< Vec3 > const& i_positions, std::vector< unsigned int > const& i_triangles) {
    glBegin(GL_TRIANGLES);
    for (unsigned int tIt = 0; tIt < i_triangles.size() / 3; ++tIt) {
        Vec3 p0 = i_positions[3 * tIt];
        Vec3 p1 = i_positions[3 * tIt + 1];
        Vec3 p2 = i_positions[3 * tIt + 2];
        Vec3 n = Vec3::cross(p1 - p0, p2 - p0);
        n.normalize();
        glNormal3f(n[0], n[1], n[2]);
        glVertex3f(p0[0], p0[1], p0[2]);
        glVertex3f(p1[0], p1[1], p1[2]);
        glVertex3f(p2[0], p2[1], p2[2]);
    }
    glEnd();
}

void drawTriangleMesh(std::vector< Vec3 > const& i_positions) {
    glBegin(GL_TRIANGLES);
    for (unsigned int tIt = 0; tIt < i_positions.size() / 3; ++tIt) {
        Vec3 p0 = i_positions[3 * tIt];
        Vec3 p1 = i_positions[3 * tIt + 1];
        Vec3 p2 = i_positions[3 * tIt + 2];
        Vec3 n = Vec3::cross(p1 - p0, p2 - p0);
        n.normalize();
        glNormal3f(n[0], n[1], n[2]);
        glVertex3f(p0[0], p0[1], p0[2]);
        glVertex3f(p1[0], p1[1], p1[2]);
        glVertex3f(p2[0], p2[1], p2[2]);
    }
    glEnd();
}

void drawPointSet(std::vector< Vec3 > const& i_positions, std::vector< Vec3 > const& i_normals) {
    glBegin(GL_POINTS);
    for (unsigned int pIt = 0; pIt < i_positions.size(); ++pIt) {
        glNormal3f(i_normals[pIt][0], i_normals[pIt][1], i_normals[pIt][2]);
        glVertex3f(i_positions[pIt][0], i_positions[pIt][1], i_positions[pIt][2]);
    }
    glEnd();
}


// Draw only modele 
void drawModele() {
    glPointSize(2); // for example...
    glColor3f(0.8, 0.8, 1);
    drawPointSet(positions, normals);
}

// Draw only projected points 
void drawPoints(std::vector<Vec3>const& projpos, std::vector<Vec3>const& projnorm,
    float nR = 1.0f, float nG = 0.5f, float nB = 0.5f, int pointsize = 4)
{

    glPointSize(pointsize);
    glColor3f(nR, nG, nB);
    drawPointSet(projpos, projnorm);
}

void draw() {
    glPointSize(2); // for example...

    // white
    if (displayModel)
    {
        glColor3f(0.8, 0.8, 1);
        drawPointSet(positions, normals);
    }

    // draw grid 
    //drawPoints(new_grid, grid_norms, 0.5, 0.5, 0.9);

    // draw sommet 
    if (displayVoxelCenter)
        drawPoints(voxelSommets, voxelSommetNorms, 1.0f, 0.5f, 0.0, 8);

    glColor3f(1.0f, 0.0, 0.0);
    if (displaySurface)
        drawTriangleMesh(surfTrig);

}



void display() {
    glLoadIdentity();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    camera.apply();
    draw();
    glFlush();
    glutSwapBuffers();
}

void idle() {
    glutPostRedisplay();
}

void key(unsigned char keyPressed, int x, int y) {
    switch (keyPressed) {
    case 'f':
        if (fullScreen == true) {
            glutReshapeWindow(SCREENWIDTH, SCREENHEIGHT);
            fullScreen = false;
        }
        else {
            glutFullScreen();
            fullScreen = true;
        }
        break;

    case 'w':
        GLint polygonMode[2];
        glGetIntegerv(GL_POLYGON_MODE, polygonMode);
        if (polygonMode[0] != GL_FILL)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        else
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        break;

    case 'm':
        displayModel = !displayModel;
        break;

    case 'g':
        displayVoxelCenter = !displayVoxelCenter;
        break;

    case 'l':
        displaySurface = !displaySurface;
        break;

    default:
        break;
    }
    idle();
}

void mouse(int button, int state, int x, int y) {
    if (state == GLUT_UP) {
        mouseMovePressed = false;
        mouseRotatePressed = false;
        mouseZoomPressed = false;
    }
    else {
        if (button == GLUT_LEFT_BUTTON) {
            camera.beginRotate(x, y);
            mouseMovePressed = false;
            mouseRotatePressed = true;
            mouseZoomPressed = false;
        }
        else if (button == GLUT_RIGHT_BUTTON) {
            lastX = x;
            lastY = y;
            mouseMovePressed = true;
            mouseRotatePressed = false;
            mouseZoomPressed = false;
        }
        else if (button == GLUT_MIDDLE_BUTTON) {
            if (mouseZoomPressed == false) {
                lastZoom = y;
                mouseMovePressed = false;
                mouseRotatePressed = false;
                mouseZoomPressed = true;
            }
        }
    }
    idle();
}

void motion(int x, int y) {
    if (mouseRotatePressed == true) {
        camera.rotate(x, y);
    }
    else if (mouseMovePressed == true) {
        camera.move((x - lastX) / static_cast<float>(SCREENWIDTH), (lastY - y) / static_cast<float>(SCREENHEIGHT), 0.0);
        lastX = x;
        lastY = y;
    }
    else if (mouseZoomPressed == true) {
        camera.zoom(float(y - lastZoom) / SCREENHEIGHT);
        lastZoom = y;
    }
}


void reshape(int w, int h) {
    camera.resize(w, h);
}


// ================================================================================================= // 
//                                    Implementation TP 
// ================================================================================================= //


/* -------------------------------------------------------------------- */
//                               TP 1 
/* ---------------------------------------------------------------------*/



/* Kernel */
double GaussianKern(double r, double h)
{
    return std::exp(-(r * r) / (h * h));
}

double WendlandKern(double r, double h)
{
    return std::pow(1.0f - (r / h), 4.0f) * (1.0f + (4.0f * r / h));
}

double SingulierKern(double r, double h, double s = 2.0)
{

    double result = std::pow((h / r), s);
    return result;
}

// x = point a projeter
// n normal
// voisin : point voisin
Vec3 projectionPlan(Vec3 const& x, Vec3 const& n, Vec3 const& voisin)
{
    Vec3 cx = x - voisin;
    Vec3 pci = x - Vec3::dot(n, cx) * n;

    return pci;
}

// Process weights wi 
double pwi(float r, float h, int kernel_type = 5)
{
    double wk = 0.0;
    // weight wi 
    if (kernel_type == 0) // gaussian 
    {
        wk = GaussianKern(r, h);
    }
    else if (kernel_type == 1)
    {
        wk = WendlandKern(r, h);
    }
    else if (kernel_type == 2)
    {
        wk = SingulierKern(r, h, 2.0f);
    }
    else
    {
        // use distance 
        wk = 1.0f / h;
    }

    return wk;
}

template<typename T>
Vec3 getPonderateMeanVec3(std::vector<Vec3> const& points, std::vector<T> const& weights)
{
    Vec3 m_point = { 0.0f, 0.0f, 0.0f };
    double sum_weights = 0.0;

    // mean of projected points
    for (unsigned int ki = 0; ki < points.size(); ki++)
    {
        m_point += weights[ki] * points[ki];
        sum_weights += weights[ki];
    }
    m_point /= sum_weights;


    return m_point;
}


void HPSS(Vec3 inputPoint, Vec3& outputPoint, Vec3& outputNormal,
    std::vector<Vec3> const& positions, std::vector<Vec3> const& normals,
    BasicANNkdTree const& kdtree, int kernel_type,
    unsigned int nbIterations = 10, unsigned int knn = 20)
{

    // definition du voisinage d'inputPoint = knn neighboohood 
    ANNidxArray id_nn = new ANNidx[knn];
    ANNdistArray square_distances_to_neighbors = new ANNdist[knn];

    // init loop data 
    Vec3 xk = inputPoint;
    Vec3 nk; // mean normal
    Vec3 ck; // mean centroid
    std::vector<Vec3> centroids;
    std::vector<Vec3> normals_k;
    std::vector<double> wi;

    for (unsigned int it = 0; it < nbIterations; it++)
    {
        // Knn nearest neighbours 
        kdtree.knearest(xk, knn, id_nn, square_distances_to_neighbors);

        // Process centroids and weights 
        centroids.clear();
        normals_k.clear();
        wi.clear();

        // Find the Nearest Neighboors 
        for (unsigned int k = 0; k < knn; k++)
        {

            Vec3 proj_xk = projectionPlan(xk, normals[id_nn[k]], positions[id_nn[k]]);
            centroids.push_back(proj_xk);
            normals_k.push_back(normals[id_nn[k]]);

            // weight wi 
            float r = (xk - positions[id_nn[k]]).length();
            float h = std::sqrt(square_distances_to_neighbors[knn - 1]);
            wi.push_back(pwi(r, h, kernel_type));
        }

        // mean ponderate projected point 
        ck = getPonderateMeanVec3(centroids, wi);

        // mean ponderate normal
        nk = getPonderateMeanVec3(normals_k, wi);
        nk.normalize(); 

        // projection
        xk = projectionPlan(xk, nk, ck);

    }

    // assign 
    outputPoint = xk;
    outputNormal = nk;

    // free memory
    delete[] id_nn;
    delete[] square_distances_to_neighbors;

}




// Add noise on the normal axis 
void AddNoise(std::vector<Vec3>& signal3D, std::vector<Vec3>const& normal3D, const float& alpha)
{

    for (unsigned int k = 0; k < signal3D.size(); k++)
    {
        float beta = (double)(rand()) / (double)(RAND_MAX)*alpha;
        signal3D[k] += beta * normal3D[k];
    }

}

// Create a random cloud of points 
void createArtificalPointset(const int sizevec, std::vector<Vec3>& new_positions, std::vector<Vec3>& new_normals)
{
    new_positions.resize(sizevec);
    new_normals.resize(new_positions.size());
    for (unsigned int pIt = 0; pIt < new_positions.size(); ++pIt) {
        new_positions[pIt] = Vec3(
            -0.6 + 1.2 * (double)(rand()) / (double)(RAND_MAX),
            -0.6 + 1.2 * (double)(rand()) / (double)(RAND_MAX),
            -0.6 + 1.2 * (double)(rand()) / (double)(RAND_MAX)
        );
    }

}

/* ---------------------------------------------------------------------------------- */
//                                      TP 2 
/* -----------------------------------------------------------------------------------*/

IMPL_VALUE implicite_func(Vec3 const& inputPoint, std::vector<Vec3>const& hpss_points, std::vector<Vec3>const& hpss_normals, BasicANNkdTree const& kdtree)
{
    Vec3 px;
    Vec3 pnx;
    IMPL_VALUE res;
    // Projection HPSS - Gaussian Kernel 20 neighb 
    HPSS(inputPoint, px, pnx, hpss_points, hpss_normals, kdtree, 0, 10, 20);

    double fx = Vec3::dot(inputPoint - px, pnx);

    if (fx > 0.0f)
    {
        res = IMPL_VALUE::EXTERNE;
    }
    else if (fx < 0.0f)
    {
        res = IMPL_VALUE::INTERNE;
    }
    else
    {
        res = IMPL_VALUE::ON_SURFACE;
    }

    return res;
}


void setBoundingBox(Vec3& bbmin, Vec3& bbmax, std::vector<Vec3>const& points)
{
    // Create a uniformly distributed pointset 
    bbmin = Vec3{ FLT_MAX, FLT_MAX, FLT_MAX };
    bbmax = Vec3{ FLT_MIN, FLT_MIN, FLT_MIN };

    // bounding box 
    for (int k = 0; k < points.size(); k++)
    {
        Vec3 point = points[k];

        if (bbmin[0] > point[0])
            bbmin[0] = point[0];

        if (bbmin[1] > point[1])
            bbmin[1] = point[1];

        if (bbmin[2] > point[2])
            bbmin[2] = point[2];

        if (bbmax[0] < point[0])
            bbmax[0] = point[0];

        if (bbmax[1] < point[1])
            bbmax[1] = point[1];

        if (bbmax[2] < point[2])
            bbmax[2] = point[2];
    }

    Vec3 delta_diag = 0.05 * (bbmax - bbmin);
    bbmax += delta_diag;
    bbmin -= delta_diag;
}

void createGridPointset(const int sizeCube,
    std::vector<Vec3>& grid, std::vector<Vec3>& gridnorms,
    std::vector<IMPL_VALUE>& grid_value,
    std::vector< std::vector< std::vector< Voxel > > >& voxelGrid,
    std::vector<Vec3>const& hpss_points, std::vector<Vec3>const& hpss_normals, BasicANNkdTree const& kdtree)
{
    Vec3 BBmin; 
    Vec3 BBmax; 
    setBoundingBox(BBmin, BBmax, hpss_points);
    
    Vec3 pasCube = (BBmax - BBmin) / float(sizeCube);
    float pasSimple = pasCube[0]; // same step on x y z 

    grid.reserve(sizeCube * sizeCube * sizeCube);
    grid_value.reserve(sizeCube * sizeCube * sizeCube);
    gridnorms.reserve(sizeCube * sizeCube * sizeCube);
    voxelGrid.reserve(sizeCube * sizeCube * sizeCube);

    for (float kx = BBmin[0]; kx < BBmax[0]; kx += pasCube[0])
    {
        std::vector<std::vector<Voxel>> vy{};
        
        for (float ky = BBmin[1]; ky < BBmax[1]; ky += pasCube[1])
        {
            std::vector<Voxel> vz{}; 
            for (float kz = BBmin[2]; kz < BBmax[2]; kz += pasCube[2])
            {
                grid.push_back(Vec3(kx, ky, kz));
                // evaluate implicite func on last point grid[-1] 
                grid_value.push_back(implicite_func(*(grid.end() - 1), hpss_points, hpss_normals, kdtree));
                gridnorms.push_back(Vec3(0, 1.0, 0.0));

                // VoxelGrid 
                vz.push_back(Voxel(kx, ky, kz, pasCube));  
            }

            vy.push_back(vz); 
          
        }

        voxelGrid.push_back(vy); 
    }
}

// Set the detection of change and list of sommets
void detectChange(int ncubes, const int nSommets, std::vector< std::vector< std::vector< Voxel > > >& voxGrid,
    std::vector<Vec3>& voxSommets, std::vector<Vec3>& voxNorms,
    std::vector<Vec3>const& pointset, std::vector<Vec3>const& pnormals, BasicANNkdTree const& kd_tree)
{

    Vec3 Up{ 1.0f, 0.0f, 0.0f };
    // Check if the voxel is on a change of interior/exterior 
    IMPL_VALUE resultImplFunc;
    int nArretes = nSommets + 4; 

    for (int kx = 0; kx < ncubes; kx++)
    {

        for (int ky = 0; ky < ncubes; ky++)
        {

            for (int kz = 0; kz < ncubes; kz++)
            {
                std::vector<IMPL_VALUE> implicite_KA = {};
                resultImplFunc = implicite_func(voxGrid[kx][ky][kz].getSommets()[0], pointset, pnormals, kd_tree);
                bool prevRes = true;
                implicite_KA.push_back(resultImplFunc);

                // check si changement de function implicite 
                for (int k = 1; k < nSommets; k++)
                {
                    IMPL_VALUE current = implicite_func(voxGrid[kx][ky][kz].getSommets()[k], pointset, pnormals, kd_tree);
                    prevRes &= (resultImplFunc == current);
                    resultImplFunc = current;
                    implicite_KA.push_back(resultImplFunc);
                }

                if (!prevRes)
                {
                    //change detect set center 
                    Vec3 center = voxGrid[kx][ky][kz].getCenter(true);
                    voxSommets.push_back(center);
                    voxNorms.push_back(Up);

                    // apply to arrete 
                    for (int ka = 0; ka < nArretes; ka++)
                    {
                        std::array<int, 2>  somA = voxGrid[kx][ky][kz].getNumberArrete(ka);
                        if (implicite_KA[somA[0]] != implicite_KA[somA[1]])
                            voxGrid[kx][ky][kz].changeArrete(ka);
                    }
                }
            }
        }
    }


}

Vec3 getMidPoint(Vec3& sA, Vec3 sB)
{
    Vec3 sC = { (sA[0] + sB[0]) / 2.0f, (sA[1] + sB[1]) / 2.0f , (sA[2] + sB[2]) / 2.0f }; 
    return sC;
}


// subdivision 
std::vector<Vec3> getMeshTrig(Vec3& s1, Vec3& s2, Vec3& s3)
{
    std::vector<Vec3> subtrig; 
    subtrig.push_back(s1); 
    subtrig.push_back(getMidPoint(s1,s2));
    subtrig.push_back(getMidPoint(s3, s1));
    
    subtrig.push_back(getMidPoint(s1, s2));
    subtrig.push_back(s2);
    subtrig.push_back(getMidPoint(s2, s3));

    subtrig.push_back(getMidPoint(s1, s2));
    subtrig.push_back(getMidPoint(s2, s3));
    subtrig.push_back(getMidPoint(s1, s3));

    subtrig.push_back(getMidPoint(s2, s3));
    subtrig.push_back(s3);
    subtrig.push_back(getMidPoint(s1, s3));

    return subtrig; 

}

void setSurfaceTrig(std::vector<Vec3>& trigdata,
    Voxel& s0, Voxel& s1, Voxel& s2, Voxel& s3,
    int methode, std::vector<Vec3> const& positions,
    std::vector<Vec3> const& normals,
    BasicANNkdTree const& kdtree, int kernel_type=0,
    unsigned int nbIterations = 10, unsigned int knn = 5)
{   
    Vec3 sommet_s0 = s0.getCenter(); 
    Vec3 sommet_s1 = s1.getCenter(); 
    Vec3 sommet_s2 = s2.getCenter(); 
    Vec3 sommet_s3 = s3.getCenter(); 

    if (methode == 0)
    {
        // point central 
        trigdata.push_back(sommet_s0);
        trigdata.push_back(sommet_s1);
        trigdata.push_back(sommet_s2);

        trigdata.push_back(sommet_s0);
        trigdata.push_back(sommet_s2);
        trigdata.push_back(sommet_s3);
    }
    else if (methode == 1)
    {
        // HPSS sur le point central 
        Vec3 center;
        Vec3 norml;
        HPSS(sommet_s0, center, norml, positions, normals, kdtree, kernel_type, nbIterations, knn);
        trigdata.push_back(center);
        HPSS(sommet_s1, center, norml, positions, normals, kdtree, kernel_type, nbIterations, knn);
        trigdata.push_back(center);
        HPSS(sommet_s2, center, norml, positions, normals, kdtree, kernel_type, nbIterations, knn);
        trigdata.push_back(center);

        HPSS(sommet_s0, center, norml, positions, normals, kdtree, kernel_type, nbIterations, knn);
        trigdata.push_back(center);
        HPSS(sommet_s2, center, norml, positions, normals, kdtree, kernel_type, nbIterations, knn);
        trigdata.push_back(center);
        HPSS(sommet_s3, center, norml, positions, normals, kdtree, kernel_type, nbIterations, knn);
        trigdata.push_back(center);
    }
    else if (methode == 2)
    {
        // Subdivision
        std::vector<Vec3> s_to_proj; 
        s_to_proj = getMeshTrig(sommet_s0, sommet_s1, sommet_s2);
        for (const Vec3& sm : s_to_proj)
        {
            Vec3 center;
            Vec3 norml;
            HPSS(sm, center, norml, positions, normals, kdtree, kernel_type, nbIterations, knn);
            trigdata.push_back(center);

        }

        s_to_proj = getMeshTrig(sommet_s0, sommet_s2, sommet_s3);
        for (const Vec3& sm : s_to_proj)
        {
            Vec3 center;
            Vec3 norml;
            HPSS(sm, center, norml, positions, normals, kdtree, kernel_type, nbIterations, knn);
            trigdata.push_back(center);

        }

    }
    else
    {
        // point central 
        trigdata.push_back(sommet_s0);
        trigdata.push_back(sommet_s1);
        trigdata.push_back(sommet_s2);

        trigdata.push_back(sommet_s0);
        trigdata.push_back(sommet_s2);
        trigdata.push_back(sommet_s3);
    }; 

}


void surfaceCreation(int ncubes, std::vector< std::vector< std::vector< Voxel > > >& voxGrid,
                    int meth, std::vector<Vec3>& surfaceTrig,  std::vector<Vec3>const& pointset, 
                    std::vector<Vec3>const& pnormals, BasicANNkdTree const& kd_tree)
{
    // Surface Creation : List of triangles 
    for (int kx = 0; kx < ncubes; kx++)
    {
        for (int ky = 0; ky < ncubes; ky++)
        {
            for (int kz = 0; kz < ncubes; kz++)
            {
                // Changement sur interieur / exterieur dans le voxel 
                if (!voxGrid[kx][ky][kz].isUsedGeom())
                {
                    continue;
                }

                Voxel vox = voxGrid[kx][ky][kz];
                std::vector<int> vec_ka = voxGrid[kx][ky][kz].changeArrete();
                for (int& kai : vec_ka)
                {
                    Voxel vvox1;
                    Voxel vvox2;
                    Voxel vvox3;

                    // defined neighb 
                    switch (kai)
                    {
                    case(0):
                        vvox1 = voxGrid[kx][ky - 1][kz];
                        vvox2 = voxGrid[kx][ky - 1][kz - 1];
                        vvox3 = voxGrid[kx][ky][kz - 1];
                        break;
                    case(1):
                        vvox1 = voxGrid[kx + 1][ky][kz];
                        vvox2 = voxGrid[kx + 1][ky][kz - 1];
                        vvox3 = voxGrid[kx][ky][kz - 1];
                        break;
                    case(2):
                        vvox1 = voxGrid[kx][ky + 1][kz];
                        vvox2 = voxGrid[kx][ky + 1][kz - 1];
                        vvox3 = voxGrid[kx][ky][kz - 1];
                        break;
                    case(3):
                        vvox1 = voxGrid[kx - 1][ky][kz];
                        vvox2 = voxGrid[kx - 1][ky][kz - 1];
                        vvox3 = voxGrid[kx][ky][kz - 1];
                        break;
                    case(4):
                        vvox1 = voxGrid[kx][ky - 1][kz];
                        vvox2 = voxGrid[kx][ky - 1][kz + 1];
                        vvox3 = voxGrid[kx][ky][kz + 1];
                        break;
                    case(5):
                        vvox1 = voxGrid[kx + 1][ky][kz];
                        vvox2 = voxGrid[kx + 1][ky][kz + 1];
                        vvox3 = voxGrid[kx][ky][kz + 1];
                        break;
                    case(6):
                        vvox1 = voxGrid[kx][ky + 1][kz];
                        vvox2 = voxGrid[kx][ky + 1][kz + 1];
                        vvox3 = voxGrid[kx][ky][kz + 1];
                        break;
                    case(7):
                        vvox1 = voxGrid[kx - 1][ky][kz];
                        vvox2 = voxGrid[kx - 1][ky][kz + 1];
                        vvox3 = voxGrid[kx][ky][kz + 1];
                        break;
                    case(8):
                        vvox1 = voxGrid[kx - 1][ky][kz];
                        vvox2 = voxGrid[kx - 1][ky - 1][kz];
                        vvox3 = voxGrid[kx][ky - 1][kz];
                        break;
                    case(9):
                        vvox1 = voxGrid[kx + 1][ky][kz];
                        vvox2 = voxGrid[kx + 1][ky - 1][kz];
                        vvox3 = voxGrid[kx][ky - 1][kz];
                        break;
                    case(10):
                        vvox1 = voxGrid[kx + 1][ky][kz];
                        vvox2 = voxGrid[kx + 1][ky + 1][kz];
                        vvox3 = voxGrid[kx][ky + 1][kz];
                        break;
                    case(11):
                        vvox1 = voxGrid[kx - 1][ky][kz];
                        vvox2 = voxGrid[kx - 1][ky + 1][kz];
                        vvox3 = voxGrid[kx][ky + 1][kz];
                        break;
                    default:
                        vvox1 = vox;
                        vvox2 = vox;
                        vvox3 = vox;
                        break;
                    }

                    setSurfaceTrig(surfaceTrig, vox, vvox1, vvox2, vvox3, meth, pointset, pnormals, kd_tree);
                }
            }
        }
    }

}



int main(int argc, char** argv)
{
    if (argc > 2) {
        exit(EXIT_FAILURE);
    }
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(SCREENWIDTH, SCREENHEIGHT);
    window = glutCreateWindow("TP HPSS APSS");

    init();
    glutIdleFunc(idle);
    glutDisplayFunc(display);
    glutKeyboardFunc(key);
    glutReshapeFunc(reshape);
    glutMotionFunc(motion);
    glutMouseFunc(mouse);
    key('?', 0, 0);

    std::cout << "Press m to remove/display model\n";
    std::cout << "Press g to remove/display Voxel Centers of voxels with change \n";
    std::cout << "Press l to remove/display Surface processed\n";
    std::cout << "Press w to activate/deactivate wireframe\n";


    {
        // Load a first pointset, and build a kd-tree:
        loadPN("data/pointsets/igea.pn", positions, normals);

        BasicANNkdTree kdtree;
        kdtree.build(positions);

        // Create A grid of voxel 
        int nCubes = 16;
        std::vector<IMPL_VALUE> voxelValue;
        std::vector< std::vector< std::vector< Voxel > > > voxelGrid;

        const int N = 8; // 8 sommets 
        createGridPointset(nCubes, new_grid, grid_norms, voxelValue, voxelGrid, positions, normals, kdtree);

        // detect interior/exterior change in the voxel Grid 
        detectChange(nCubes, N, voxelGrid, voxelSommets, voxelSommetNorms, positions, normals, kdtree);

        // -----------------------------------------------------
        // Surface Créée à partir de : 
        // ---------------------------
        // 0 CENTRAL = point central dans le voxel (création de 2 triangles) pour un changement d'arrête
        // 1 CENTRAL_PROJ Proj = points des triangles projetés sur le nuage de points à l'aide de l'algo HPSS 
        // 2 SUBSURF_PROJ = Ajout d'une subdivision supplémentaire et projection HPSS (création de 8 triangles)

        enum SurfType { CENTRAL = 0, CENTRAL_PROJ = 1, SUBSURF_PROJ = 2 };
        surfaceCreation(nCubes, voxelGrid, (int)SurfType::SUBSURF_PROJ, surfTrig, positions, normals, kdtree);
     

        glutMainLoop();
        return EXIT_SUCCESS;
    }
}
