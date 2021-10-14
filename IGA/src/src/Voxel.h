#pragma once
#include <vector>
#include <array>
#include "Vec3.h"

// Placer ce fichier dans le dossier src 
// au meme niveau que Vec3.h 

class Arrete
{
private: 
    Vec3 m_sommet1; 
    Vec3 m_sommet2; 
public: 
    Arrete(const Vec3& s1, const Vec3 s2)
        : m_sommet1(s1), m_sommet2(s2)
    {}; 

    ~Arrete() {};

    // Copy ctor 
    Arrete(const Arrete& source)
    {
        m_sommet1 = source.m_sommet1; 
        m_sommet2 = source.m_sommet2;
    }

    // 3 : copy assignment operator
    Arrete& operator=(const Arrete& source) 
    {
        if (this == &source)
            return *this;

        m_sommet1 = source.m_sommet1;
        m_sommet2 = source.m_sommet2;

        return *this;

    }

    // 4 : move constructor
    Arrete(Arrete&& source) 
    {
        m_sommet1 = source.m_sommet1;
        m_sommet2 = source.m_sommet2;

        source.m_sommet1 = Vec3(0.0f, 0.0f, 0.0f); 
        source.m_sommet2 = Vec3(0.0f, 0.0f, 0.0f);
    }

    bool operator==(Arrete& source)
    {
        if (source.m_sommet1 != m_sommet1 || source.m_sommet2 != m_sommet2)
            return true; 
        
        
        return false; 
    }



};


class Voxel
{
private:
    Vec3 m_pos;
    float m_pas;
    Vec3 m_pas3D; 
    int m_id = -1;
    std::vector<Vec3> m_sommets;
    std::vector<Vec3> m_normals;

    Vec3 m_center{ 0.0, 0.0, 0.0 };
    bool usedInGeometry = false; 
    std::vector<int> m_change; 

public:

    static int id;



    Voxel()
    : m_pos(Vec3{0.0f, 0.0f, 0.0f}), m_pas(-1.0f), m_pas3D(Vec3{ -1.0f ,-1.0f ,-1.0f })
    {}; 

    // Ctor
    Voxel(float x, float y, float z, float pas)
        : m_pas(pas)
    {
        // lower left position 
        m_pos = Vec3(x, y, z);
        m_sommets.resize(8); 
        m_normals.resize(8);
        initVoxel();
        Center();
        m_id = id;
        id++;
    };

    // Ctor
    Voxel(float x, float y, float z, Vec3 pas)
        : m_pas(-1.0f), m_pas3D(pas)
    {
        // lower left position 
        m_pos = Vec3(x, y, z);
        m_sommets.resize(8);
        m_normals.resize(8);
        initVoxel();
        Center();
        m_id = id;
        id++;
    };

    ~Voxel() {};

    // Copy ctor 
    Voxel(const Voxel& source)
    {
        m_pos = source.m_pos;
        m_pas = source.m_pas; 
        m_pas3D = source.m_pas3D; 
        m_id = source.m_id; 
        m_sommets = source.m_sommets;
        m_normals = source.m_normals; 
        m_center = source.m_center;
        m_change = source.m_change;
    }

    // Copy Assignment 
    Voxel& operator=(const Voxel& source)
    {
        if (this == &source)
            return *this; 

        m_pos = source.m_pos;
        m_pas = source.m_pas;
        m_pas3D = source.m_pas3D;
        m_id = source.m_id;

        m_change.clear(); 
        m_change.resize(source.m_change.size());
        m_change = source.m_change;

        m_sommets.clear(); 
        m_sommets.resize(source.m_sommets.size()); 
        m_sommets = source.m_sommets;

        m_normals.clear(); 
        m_normals.resize(source.m_normals.size());
        m_normals = source.m_normals;

        m_center = source.m_center;

        return *this; 

    }

    // Moving Ctor 
    Voxel(Voxel&& source) 
    {
        //std::cout << "Moving Instance " << source.m_id << std::endl; 
        m_pos = source.m_pos;
        m_pas = source.m_pas;
        m_pas3D = source.m_pas3D;
        m_id = source.m_id;
         
        m_change.clear();
        m_change.resize(source.m_change.size());
        m_change = source.m_change;

        m_sommets.clear();
        m_sommets.resize(source.m_sommets.size());
        m_sommets = source.m_sommets;

        m_normals.clear();
        m_normals.resize(source.m_normals.size());
        m_normals = source.m_normals;

        m_center = source.m_center; 

        // invalidate source
        source.m_pos = Vec3{ 0.0f, 0.0f, 0.0f };
        source.m_pas = -1.0f;
        source.m_pas3D = Vec3{ -1.0f, -1.0f, -1.0f };
        source.m_id = -1;
        source.m_center = Vec3{ 0.0f, 0.0f, 0.0f };
        source.m_sommets.clear(); 
        source.m_sommets.resize(0); 
        source.m_normals.clear(); 
        source.m_normals.resize(0); 
        source.m_change.clear(); 
        source.m_change.resize(0);

    }

    void initVoxel()
    {
        m_sommets.clear();
        if (m_pas > 0.0f)
        {
            m_sommets.push_back(this->m_pos);                       // 0
            m_sommets.push_back(m_pos + Vec3(m_pas, 0.0, 0.0));     // 1 
            m_sommets.push_back(m_pos + Vec3(m_pas, m_pas, 0.0));   // 2
            m_sommets.push_back(m_pos + Vec3(0.0, m_pas, 0.0));     // 3

            m_sommets.push_back(m_pos + Vec3(0.0, 0.0, m_pas));     // 4
            m_sommets.push_back(m_pos + Vec3(m_pas, 0.0, m_pas));   // 5
            m_sommets.push_back(m_pos + Vec3(m_pas, m_pas, m_pas)); // 6
            m_sommets.push_back(m_pos + Vec3(0.0, m_pas, m_pas));   // 7
        }
        else
        {
            m_sommets.push_back(this->m_pos);                                   // 0
            m_sommets.push_back(m_pos + Vec3(m_pas3D[0], 0.0, 0.0));            // 1
            m_sommets.push_back(m_pos + Vec3(m_pas3D[0], m_pas3D[1], 0.0));     // 2
            m_sommets.push_back(m_pos + Vec3(0.0, m_pas3D[1], 0.0));            // 3
            
            m_sommets.push_back(m_pos + Vec3(0.0, 0.0, m_pas3D[2]));            // 4
            m_sommets.push_back(m_pos + Vec3(m_pas3D[0], 0.0, m_pas3D[2]));     // 5
            m_sommets.push_back(m_pos + m_pas3D);                               // 6
            m_sommets.push_back(m_pos + Vec3(0.0, m_pas3D[1], m_pas3D[2]));     // 7
        }
        

        for (int k = 0; k < 8; k++)
        {
            m_normals.push_back(Vec3{ 0.0f, 0.0f, 1.0f });
        }

    };

    void Center()
    {
        m_center[0] = 0.0f;
        m_center[1] = 0.0f;
        m_center[2] = 0.0f;

        for (int k = 0; k < 8; k++)
            m_center += m_sommets[k];

        m_center /= 8.0f;
    }

    Vec3 getCenter(bool inGeometry=false) 
    { 
        if (inGeometry)
            usedInGeometry = true; 
        
        return m_center;  
    }

    int getId() const { return m_id; }

    std::vector<Vec3> getSommets() const { return m_sommets;  }

    bool isUsedGeom() const { return usedInGeometry; }

    void changeArrete(int no) { m_change.push_back(no);  }

    std::vector<int> changeArrete() const { return m_change; }


    // x : 0 2 4 6
    // y: 1 3 5 7
    // z : 8 9 10 11
    std::array<Vec3, 2> getArrete(int no)
    {
        std::array<Vec3, 2> result; 
        switch (no)
        {
        case 0:
            result = { m_sommets[0], m_sommets[1] };
            break;
        case 1:
            result = { m_sommets[1], m_sommets[2] };
            break;
        case 2:
            result = { m_sommets[2], m_sommets[3] };
            break;
        case 3:
            result = { m_sommets[3], m_sommets[0] };
            break;
        case 4:
            result = { m_sommets[4], m_sommets[5] };
            break;
        case 5:
            result = { m_sommets[5], m_sommets[6] };
            break;
        case 6:
            result = { m_sommets[6], m_sommets[7] };
            break;
        case 7:
            result = { m_sommets[7], m_sommets[4] };
            break;
        case 8:
            result = { m_sommets[0], m_sommets[4] };
            break;
        case 9:
            result = { m_sommets[1], m_sommets[5] };
            break;
        case 10:
            result = { m_sommets[2], m_sommets[6] };
            break;
        case 11:
            result = { m_sommets[3], m_sommets[7] };
            break;
        default: 
            result = { m_sommets[0], m_sommets[1] }; 
            break;
        };
        return result; 
    }

    std::array<int,2> getNumberArrete(int no)
    {
        std::array<int, 2> result;
        switch (no)
        {
        case 0:
            result = {0, 1};
            break;
        case 1:
            result = {1, 2};
            break;
        case 2:
            result = {2,3 };
            break;
        case 3:
            result = {3, 0};
            break;
        case 4:
            result = {4,5};
            break;
        case 5:
            result = {5,6};
            break;
        case 6:
            result = {6,7};
            break;
        case 7:
            result = {7,4};
            break;
        case 8:
            result = {0,4};
            break;
        case 9:
            result = {1,5};
            break;
        case 10:
            result = {2,6};
            break;
        case 11:
            result = {3,7};
            break;
        default:
            result = {0,1 };
            break;
        };
        return result;
    }
};