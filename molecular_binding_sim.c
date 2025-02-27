// 2 - High performance C implementation of molecular binding simulation

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Molecular structure definitions
typedef struct {
    double x, y, z;
} Vec3;

typedef struct {
    int atom_type;   // Element type (1=H, 6=C, 7=N, 8=O, etc.)
    Vec3 position;   // Position in 3D space
    double charge;   // Partial charge
    double radius;   // van der Waals radius
    int bonded_to[8]; // Indices of bonded atoms (max 8)
    int num_bonds;   // Number of bonds
} Atom;

typedef struct {
    Atom* atoms;     // Array of atoms
    int num_atoms;   // Number of atoms
    char name[64];   // Molecule name
    double energy;   // Current energy
    Vec3 center;     // Center of mass
    double* tension_field; // Pointer to external tension field
    int field_resolution;  // Resolution of tension field
} Molecule;

// String-Cube Interaction Functions

// Calculate binding energy between two molecules based on:
// 1. van der Waals interactions
// 2. Electrostatic interactions
// 3. Hydrogen bonding
// 4. Tension field influence
double calculate_binding_energy(const Molecule* mol1, const Molecule* mol2) {
    double energy = 0.0;
    
    // Loop over all atom pairs
    for (int i = 0; i < mol1->num_atoms; i++) {
        for (int j = 0; j < mol2->num_atoms; j++) {
            const Atom* a1 = &mol1->atoms[i];
            const Atom* a2 = &mol2->atoms[j];
            
            // Calculate distance
            double dx = a1->position.x - a2->position.x;
            double dy = a1->position.y - a2->position.y;
            double dz = a1->position.z - a2->position.z;
            double r2 = dx*dx + dy*dy + dz*dz;
            double r = sqrt(r2);
            
            // Skip atoms that are too far apart
            if (r > 12.0) continue;
            
            // van der Waals (Lennard-Jones potential)
            double sigma = (a1->radius + a2->radius) / 2.0;
            double epsilon = 0.1; // kcal/mol - typical value
            double ratio = sigma / r;
            double ratio6 = ratio * ratio * ratio * ratio * ratio * ratio;
            double ratio12 = ratio6 * ratio6;
            double vdw = epsilon * (ratio12 - 2.0 * ratio6);
            
            // Electrostatic (Coulomb potential)
            // Units: 332.0636 kcal·mol−1·Å converts to kcal/mol
            double kElectrostatic = 332.0636; 
            double electrostatic = kElectrostatic * a1->charge * a2->charge / r;
            
            // Hydrogen bonding
            // Simple model: H-bonds occur between H-X pairs where X is O or N
            double hbond = 0.0;
            if ((a1->atom_type == 1 && (a2->atom_type == 7 || a2->atom_type == 8)) ||
                (a2->atom_type == 1 && (a1->atom_type == 7 || a1->atom_type == 8))) {
                // Check distance criteria (H-bonds typically 1.5-2.5 Å)
                if (r >= 1.5 && r <= 2.5) {
                    // Simple H-bond energy contribution (-2 to -5 kcal/mol)
                    hbond = -3.0;
                }
            }
            
            // Add to total energy
            energy += vdw + electrostatic + hbond;
        }
    }
    
    // Add tension field influence
    energy += calculate_tension_field_effect(mol1, mol2);
    
    return energy;
}

// Calculate effect of external tension field on binding
double calculate_tension_field_effect(const Molecule* mol1, const Molecule* mol2) {
    // Check if tension field exists
    if (!mol1->tension_field || !mol2->tension_field || mol1->field_resolution <= 0) {
        return 0.0;
    }
    
    // Get centers of molecules
    Vec3 center1 = mol1->center;
    Vec3 center2 = mol2->center;
    
    // Calculate tension at each center by mapping to grid
    int res = mol1->field_resolution;
    
    // Convert coordinates to grid indices (assume [-1,1] to [0,res-1] mapping)
    int x1 = (int)((center1.x + 1.0) * 0.5 * (res - 1));
    int y1 = (int)((center1.y + 1.0) * 0.5 * (res - 1));
    int z1 = (int)((center1.z + 1.0) * 0.5 * (res - 1));
    
    int x2 = (int)((center2.x + 1.0) * 0.5 * (res - 1));
    int y2 = (int)((center2.y + 1.0) * 0.5 * (res - 1));
    int z2 = (int)((center2.z + 1.0) * 0.5 * (res - 1));
    
    // Clamp to valid range
    x1 = (x1 < 0) ? 0 : ((x1 >= res) ? res - 1 : x1);
    y1 = (y1 < 0) ? 0 : ((y1 >= res) ? res - 1 : y1);
    z1 = (z1 < 0) ? 0 : ((z1 >= res) ? res - 1 : z1);
    
    x2 = (x2 < 0) ? 0 : ((x2 >= res) ? res - 1 : x2);
    y2 = (y2 < 0) ? 0 : ((y2 >= res) ? res - 1 : y2);
    z2 = (z2 < 0) ? 0 : ((z2 >= res) ? res - 1 : z2);
    
    // Get tension values
    double tension1 = mol1->tension_field[x1 + y1*res + z1*res*res];
    double tension2 = mol1->tension_field[x2 + y2*res + z2*res*res];
    
    // Calculate tension effect
    // Higher tension reduces binding energy (makes binding more favorable)
    double avg_tension = (tension1 + tension2) / 2.0;
    return -10.0 * avg_tension; // Scale factor can be adjusted
}

// Compute optimal binding conformation
void optimize_binding_conformation(Molecule* mol1, Molecule* mol2, int max_iterations) {
    // Initial energy
    double energy = calculate_binding_energy(mol1, mol2);
    
    // Monte Carlo optimization
    double temperature = 10.0; // Initial temperature
    double cooling_rate = 0.95; // Cooling rate
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // Create backup of current positions
        Vec3* backup = (Vec3*)malloc(mol2->num_atoms * sizeof(Vec3));
        for (int i = 0; i < mol2->num_atoms; i++) {
            backup[i] = mol2->atoms[i].position;
        }
        
        // Apply random rotation and translation to mol2
        double angle = (double)rand() / RAND_MAX * 0.1; // Small random angle
        double axis_x = (double)rand() / RAND_MAX * 2.0 - 1.0;
        double axis_y = (double)rand() / RAND_MAX * 2.0 - 1.0;
        double axis_z = (double)rand() / RAND_MAX * 2.0 - 1.0;
        double norm = sqrt(axis_x*axis_x + axis_y*axis_y + axis_z*axis_z);
        axis_x /= norm;
        axis_y /= norm;
        axis_z /= norm;
        
        // Translation amount
        double trans_x = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.5;
        double trans_y = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.5;
        double trans_z = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.5;
        
        // Apply transformation
        for (int i = 0; i < mol2->num_atoms; i++) {
            // Translate to origin
            double x = mol2->atoms[i].position.x - mol2->center.x;
            double y = mol2->atoms[i].position.y - mol2->center.y;
            double z = mol2->atoms[i].position.z - mol2->center.z;
            
            // Rotate
            // Quaternion rotation around arbitrary axis
            double s = sin(angle / 2);
            double c = cos(angle / 2);
            double qx = axis_x * s;
            double qy = axis_y * s;
            double qz = axis_z * s;
            double qw = c;
            
            double x_new = x * (1 - 2*qy*qy - 2*qz*qz) + y * (2*qx*qy - 2*qz*qw) + z * (2*qx*qz + 2*qy*qw);
            double y_new = x * (2*qx*qy + 2*qz*qw) + y * (1 - 2*qx*qx - 2*qz*qz) + z * (2*qy*qz - 2*qx*qw);
            double z_new = x * (2*qx*qz - 2*qy*qw) + y * (2*qy*qz + 2*qx*qw) + z * (1 - 2*qx*qx - 2*qy*qy);
            
            // Translate back and apply additional translation
            mol2->atoms[i].position.x = x_new + mol2->center.x + trans_x;
            mol2->atoms[i].position.y = y_new + mol2->center.y + trans_y;
            mol2->atoms[i].position.z = z_new + mol2->center.z + trans_z;
        }
        
        // Recalculate center of molecule
        mol2->center.x += trans_x;
        mol2->center.y += trans_y;
        mol2->center.z += trans_z;
        
        // Calculate new energy
        double new_energy = calculate_binding_energy(mol1, mol2);
        
        // Accept or reject based on Metropolis criterion
        double delta_energy = new_energy - energy;
        if (delta_energy < 0 || exp(-delta_energy / temperature) > (double)rand() / RAND_MAX) {
            // Accept new conformation
            energy = new_energy;
        } else {
            // Reject and restore previous positions
            for (int i = 0; i < mol2->num_atoms; i++) {
                mol2->atoms[i].position = backup[i];
            }
            // Restore center
            mol2->center.x -= trans_x;
            mol2->center.y -= trans_y;
            mol2->center.z -= trans_z;
        }
        
        free(backup);
        
        // Cool the system
        temperature *= cooling_rate;
    }
    
    // Update final energy
    mol2->energy = energy;
}

// Python interface functions
void init_molecule(Molecule* mol, int num_atoms) {
    mol->atoms = (Atom*)malloc(num_atoms * sizeof(Atom));
    mol->num_atoms = num_atoms;
    mol->energy = 0.0;
    mol->center.x = mol->center.y = mol->center.z = 0.0;
    mol->tension_field = NULL;
    mol->field_resolution = 0;
}

void free_molecule(Molecule* mol) {
    if (mol->atoms) {
        free(mol->atoms);
        mol->atoms = NULL;
    }
}

void set_atom(Molecule* mol, int idx, int atom_type, double x, double y, double z, 
             double charge, double radius) {
    if (idx >= 0 && idx < mol->num_atoms) {
        mol->atoms[idx].atom_type = atom_type;
        mol->atoms[idx].position.x = x;
        mol->atoms[idx].position.y = y;
        mol->atoms[idx].position.z = z;
        mol->atoms[idx].charge = charge;
        mol->atoms[idx].radius = radius;
        mol->atoms[idx].num_bonds = 0;
    }
}

void add_bond(Molecule* mol, int atom1_idx, int atom2_idx) {
    if (atom1_idx >= 0 && atom1_idx < mol->num_atoms && 
        atom2_idx >= 0 && atom2_idx < mol->num_atoms) {
        
        // Add bond to atom1
        if (mol->atoms[atom1_idx].num_bonds < 8) {
            mol->atoms[atom1_idx].bonded_to[mol->atoms[atom1_idx].num_bonds++] = atom2_idx;
        }
        
        // Add bond to atom2
        if (mol->atoms[atom2_idx].num_bonds < 8) {
            mol->atoms[atom2_idx].bonded_to[mol->atoms[atom2_idx].num_bonds++] = atom1_idx;
        }
    }
}

void set_tension_field(Molecule* mol, double* field, int resolution) {
    mol->tension_field = field;
    mol->field_resolution = resolution;
}

void calculate_molecule_center(Molecule* mol) {
    mol->center.x = mol->center.y = mol->center.z = 0.0;
    
    for (int i = 0; i < mol->num_atoms; i++) {
        mol->center.x += mol->atoms[i].position.x;
        mol->center.y += mol->atoms[i].position.y;
        mol->center.z += mol->atoms[i].position.z;
    }
    
    if (mol->num_atoms > 0) {
        mol->center.x /= mol->num_atoms;
        mol->center.y /= mol->num_atoms;
        mol->center.z /= mol->num_atoms;
    }
}
