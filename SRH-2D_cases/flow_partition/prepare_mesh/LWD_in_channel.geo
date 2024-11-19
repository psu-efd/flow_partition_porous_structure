// Gmsh geometry file for a rectangular channel with structured mesh

// Define the dimensions of the rectangle (length and height)
//background water depth h0
h0 = 1.0;    //background water depth
L0 = 0.2;    //LWD length in streamwise direction

L1 = h0*50;            //domain length 1
L2 = L0;               //domain length 2
L3 = h0*25;            //domain length 3

// Resolution
refine = 4;

n1 = refine*50;
n2 = refine*2;
n3 = refine*25;
n4 = refine*10;


//channel width factor: channel width = W_factor*h0
W_factor = 10;

//channel width
W = W_factor*h0;

lc = 0.1; //

// Define the corner points of the rectangle
Point(1) = {-L1-L0/2, 0, 0, lc};     
Point(2) = {-L0/2, 0, 0, lc};     
Point(3) = {0, W/2, 0, lc};     
Point(4) = {L0/2, 0, 0, lc};     
Point(5) = {L0/2+L3, 0, 0, lc};
Point(6) = {L0/2+L3, W, 0, lc};
Point(7) = {L0/2,    W, 0, lc};
Point(8) = {0,       W, 0, lc};
Point(9) = {-L0/2,   W, 0, lc};
Point(10) = {-L1-L0/2, W, 0, lc};


// Define the lines that form the boundary of the rectangle
Line(1) = {1, 2};  
Line(2) = {2, 4};  
Line(3) = {4, 5};  
Line(4) = {5, 6};  
Line(5) = {6, 7};
Line(6) = {7, 9};
Line(7) = {9, 10};
Line(8) = {10, 1};
Line(9) = {2, 9};
Line(10) = {7, 4};

// Define the line loop and surface of the rectangle
Line Loop(1) = {1, 9, 7, 8};
Line Loop(2) = {2, -10, 6, -9};
Line Loop(3) = {3, 4, 5, 10};
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};

//Line{10} In Surface {2};

prog1 = 1.012;

// Apply transfinite meshing to create a structured mesh
Transfinite Line {1} = n1+1 Using Progression 1.0/prog1;  
Transfinite Line {9} = n4+1;
Transfinite Line {7} = n1+1 Using Progression prog1;  
Transfinite Line {8} = n4+1;
Transfinite Surface {1};        // Structured mesh for the surface

Transfinite Line {2} = n2+1;  
Transfinite Line {10} = n4+1;
Transfinite Line {6} = n2+1;  
Transfinite Line {9} = n4+1;
Transfinite Surface {2};        // Structured mesh for the surface

prog2 = 1.025;

Transfinite Line {3} = n3+1 Using Progression prog2;  
Transfinite Line {4} = n4+1;
Transfinite Line {5} = n3+1 Using Progression 1.0/prog2;  
Transfinite Line {10} = n4+1;
Transfinite Surface {3};        // Structured mesh for the surface

// Apply recombination to generate quadrilateral elements
Recombine Surface {1};
Recombine Surface {2};
Recombine Surface {3};

// Define physical groups (for boundary conditions and regions)
//Physical Line("Monitoring_Line_1") = {10};   
Physical Line("Inlet") = {4};   
Physical Line("Outlet") = {8};  
//Physical Line("Walls") = {1, 2, 3, 5, 6, 7};  
Physical Surface("Channel") = {1,2,3};  // Entire domain

// Mesh generation settings (optional)
Mesh.Algorithm = 5;  // Structured mesh algorithm
Mesh.Format = 1;     // Export mesh in MSH format version 2.2
