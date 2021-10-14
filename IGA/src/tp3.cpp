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
#include <time.h>

#include <numeric>
#include <GL/glut.h>
#include <float.h>
#include "src/Vec3.h"
#include "src/Camera.h"
#include "src/jmkdtree.h"

// Source, Source t=0, Target
std::vector< Vec3 > positions;
std::vector< Vec3 > normals;

std::vector< Vec3 > positions2;
std::vector< Vec3 > normals2;

std::vector< Vec3 > positions3;
std::vector< Vec3 > normals3;

// ICP  
Vec3 translation_deduite; 
Mat3 rotation_deduite; 

BasicANNkdTree kdtree;
int totIterations = 0; 

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
static bool displayScatterCloud = true;
static bool displayInitScatterCloud = false;
static bool displaySingular = false;




// ------------------------------------------------------------------------------------------------------------
// Base Code : i/o and some stuff
// ------------------------------------------------------------------------------------------------------------
void loadPN(const std::string& filename, std::vector< Vec3 >& o_positions, std::vector< Vec3 >& o_normals) {
    unsigned int surfelSize = 6;
    FILE* in; 
    fopen_s(&in,filename.c_str(), "rb");
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
    fopen_s(&outfile,  filename.c_str(), "wb");
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
//  Implementation HPSS  
// ------------------------------------------------------------------------------------------------------------

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


Vec3 getMeanVec3(std::vector<Vec3> const& points)
{
    Vec3 m_point = { 0.0f, 0.0f, 0.0f };
    double sum_weights = 0.0;

    // mean of projected points
    for (unsigned int ki = 0; ki < points.size(); ki++)
    {
        m_point += points[ki];
    }
    m_point /= points.size();


    return m_point;
}

void HPSS(Vec3 inputPoint, Vec3& outputPoint, Vec3& outputNormal,
    std::vector<Vec3> const& positions, std::vector<Vec3> const& normals,
    BasicANNkdTree const& kdtree, int kernel_type,
    unsigned int nbIterations = 10, unsigned int knn = 20)
{

    // definition du voisinage � inputPoint = knn neighboohood 
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



/* ---------------------------------------------------------------------------------- */
//                                      ICP
/* -----------------------------------------------------------------------------------*/
// For the purpose of this assignement we define all the functions needed 
// before defining rendering 


// Root mean Squared Distance with target point = nearest point 
float RMSD(std::vector<Vec3>const & source, const std::vector<Vec3>const& target, BasicANNkdTree const& targetKdTree)
{
    double result = 0.0; 

    for (unsigned int k = 0; k < target.size(); k++ )
    {
        int knn = targetKdTree.nearest(source[k]); 
        result += std::pow(target[knn][0] - source[k][0], 2) +
                    std::pow(target[knn][1] - source[k][1], 2) +
                    std::pow(target[knn][2] - source[k][2], 2);
    }

    result = std::sqrt(result / static_cast<float>(target.size()));

    return result; 
}


void ICP(std::vector<Vec3>& ps, std::vector<Vec3>& nps,
        std::vector<Vec3> const& qs, std::vector<Vec3> const& nqs, 
        BasicANNkdTree const& qsKdTree, Mat3& rotation, Vec3& translation, unsigned int nbIterations=0)
{
    // centroid 
    Vec3 cT = getMeanVec3(qs);

    for (unsigned int kIt = 0; kIt < nbIterations ; kIt ++)
    {
        totIterations++; 

        // Covariance Matrice = 0.0f
        Mat3 MatCovariancePsQs = Mat3(); 

        // centroid 
        Vec3 cS = getMeanVec3(ps); 

        // Process Covariance Matrix to apply SVD 
        for (unsigned int k=0; k < ps.size(); k++)
        {
            // HPSS on point psi, take the nearest point of the projected point on the modele 
            /*Vec3 oPoint, oNormal; 
            HPSS(ps[k], oPoint, oNormal, qs, nqs, qsKdTree, 0, 10, 20);
            int knn = qsKdTree.nearest(oPoint);*/
            
            // Each point is coupled with a point from target pointset
            int knn = qsKdTree.nearest(ps[k]);
            
            MatCovariancePsQs(0,0) = MatCovariancePsQs(0,0) +  (ps[k][0] - cS[0])* (qs[knn][0]- cT[0]); 
            MatCovariancePsQs(0,1) = MatCovariancePsQs(0,1) +  (ps[k][0] - cS[0])* (qs[knn][1]- cT[1]); 
            MatCovariancePsQs(0,2) = MatCovariancePsQs(0,2) +  (ps[k][0] - cS[0])* (qs[knn][2]- cT[2]); 
            MatCovariancePsQs(1,0) = MatCovariancePsQs(1,0) +  (ps[k][1] - cS[1])* (qs[knn][0]- cT[0]); 
            MatCovariancePsQs(1,1) = MatCovariancePsQs(1,1) +  (ps[k][1] - cS[1])* (qs[knn][1]- cT[1]); 
            MatCovariancePsQs(1,2) = MatCovariancePsQs(1,2) +  (ps[k][1] - cS[1])* (qs[knn][2]- cT[2]); 
            MatCovariancePsQs(2,0) = MatCovariancePsQs(2,0) +  (ps[k][2] - cS[2])* (qs[knn][0]- cT[0]); 
            MatCovariancePsQs(2,1) = MatCovariancePsQs(2,1) +  (ps[k][2] - cS[2])* (qs[knn][1]- cT[1]); 
            MatCovariancePsQs(2,2) = MatCovariancePsQs(2,2) +  (ps[k][2] - cS[2])* (qs[knn][2]- cT[2]); 

        } 

        // -------- SVD --------- //  
        Mat3 U; 
        Mat3 Vt; 
        float sx, sy, sz; 
        MatCovariancePsQs.SVD(U, sx, sy, sz, Vt);

        Mat3 V = Vt.getTranspose();

        // Matrice Rotation = V.Ut
        rotation = V * U.getTranspose(); 

        // apply rotation
        for (unsigned int k=0; k < ps.size(); k++)
        {
            ps[k] = cT + rotation * (ps[k] - cS); 
        } 

        // convergence criteria - Not apply here use 'S' to keep iterate 
        double res = RMSD(ps, qs, qsKdTree);

        std::cout << "===========================\n";
        std::cout << "Iteration " << totIterations << std::endl;
        std::cout << "RMSD :" << res << std::endl;
    }

    
}
    

// ------------------------------------------------------------------------------------------------------------
// Base Code : rendering.
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
    glPointSize(2); 
    glColor3f(0.8, 0.8, 1);
    drawPointSet(positions, normals);
}

// Wrapper color and size for pointset
void drawPoints(std::vector<Vec3>const& projpos, std::vector<Vec3>const& projnorm,
    float nR = 1.0f, float nG = 0.5f, float nB = 0.5f, int pointsize = 4)
{
    glPointSize(pointsize);
    glColor3f(nR, nG, nB);
    drawPointSet(projpos, projnorm);
}


// display/remove  Several Pointset 
void draw() {
    glPointSize(2); // for example...

    // white = Pointset 1 = Model= Target 
    if (displayModel)
    {
        glColor3f(0.8, 0.8, 1);
        drawPointSet(positions, normals);
    }

    // red = pointset2 source 
    if (displayScatterCloud)
        drawPoints(positions2, normals2);

    // green = inital position of pointset2 that is the source 
    if (displayInitScatterCloud)
        drawPoints(positions3, normals3, 0.0, 0.8, 0.3);

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
        displayScatterCloud = !displayScatterCloud;
        break;

    case 'l':
        displayInitScatterCloud = !displayInitScatterCloud;
        break;

    case 's':
        ICP(positions2, normals2, positions, normals, kdtree, rotation_deduite, translation_deduite, 1); 
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


// =======================================================================================


int main(int argc, char** argv) {
    if (argc > 2) {
        exit(EXIT_FAILURE);
    }
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(SCREENWIDTH, SCREENHEIGHT);
    window = glutCreateWindow("Recalage de nuages de points - Implementation de l'ICP");

    init();
    glutIdleFunc(idle);
    glutDisplayFunc(display);
    glutKeyboardFunc(key);
    glutReshapeFunc(reshape);
    glutMotionFunc(motion);
    glutMouseFunc(mouse);
    key('?', 0, 0);

    std::cout << "Implementation de l'Iterative Closest Point\n"; 
    std::cout << "Press m to remove/display model\n";
    std::cout << "Press g to remove/display pointset source modified\n";
    std::cout << "Press l to remove/display pointset source original\n";
    std::cout << "Press s to launch one ICP iteration\n";


    {
        srand(time(NULL)); 
        
        // Load a first pointset, and build a kd-tree:
        loadPN("data/pointsets/dino2.pn", positions, normals);
        //loadPN("data/pointsets/dino_subsampled_extreme.pn", positions2, normals2);

        // kdtree on modele 
        kdtree.build(positions);


        // -----------------------------------------------------  
        // Simulate a second pointset that is rotated & translated
        // ----------------------------------------------------- 
        positions2 = positions; 
        normals2 = normals; 

        Mat3 ICProtation = Mat3::RandRotation(); 
        Vec3 ICPtranslation = Vec3(float(-1.0f + 2.0 * ((double)(rand())) / (double)(RAND_MAX)),
                              float(-1.0f + 2.0 * ((double)(rand())) / (double)(RAND_MAX)), 
                              float(-1.0f + 2.0 * ((double)(rand())) / (double)(RAND_MAX))); 
        
        for (unsigned int pIt = 0; pIt < positions2.size(); pIt++)
        {
            positions2[pIt] = ICProtation * positions2[pIt] + ICPtranslation; 
            normals2[pIt] = ICProtation * normals2[pIt]; 
        }


        // Initial Pointset saved in P3
        positions3 = positions2;
        normals3 = normals2;

        

        // ----------  Initialisation -------------- // 
        Vec3 deduced_translation;
        Mat3 deduced_rotation;

        // calcul centroid on ps and qs 
        Vec3 centroid_ps = getMeanVec3(positions2); 
        Vec3 centroid_qs = getMeanVec3(positions); 

        // translation of ps to qs 
        deduced_translation = centroid_qs - centroid_ps;

        // rotation 
        deduced_rotation = Mat3::RandRotation();

        for (unsigned int k=0; k < positions2.size(); k++)
        {
            positions2[k] += deduced_translation;
        } 

        // -------------------------------------- // 
        // Iterative Closest Point Algorithm
        ICP(positions2, normals2, positions, normals, kdtree, 
            deduced_rotation, deduced_translation, 1);

    }


    glutMainLoop(); 
    return EXIT_SUCCESS; 
}
