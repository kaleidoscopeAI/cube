import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any
import hashlib
import time

class QuantumStringCube:
    def __init__(self, dimensions: int = 4, resolution: int = 64, qubit_depth: int = 10):
        self.dimensions = dimensions
        self.resolution = resolution
        self.qubit_depth = qubit_depth
        
        # Initialize tensor fields - each dimension has a full tensor field
        self.tensor_fields = np.zeros([resolution] * dimensions + [dimensions, dimensions])
        self.energy_grid = np.zeros([resolution] * dimensions)
        self.quantum_phase_grid = np.zeros([resolution] * dimensions, dtype=np.complex128)
        
        # String network parameters
        self.tension_strength = 0.85
        self.entanglement_factor = 0.73
        self.string_elasticity = 0.42
        self.harmonic_damping = 0.95
        
        # Initialize quantum state
        self.state_vector = self._initialize_quantum_state()
        
        # Graph representation of the cube's structure
        self.node_graph = nx.Graph()
        self.edge_tensions = {}
        
    def _initialize_quantum_state(self) -> np.ndarray:
        """Initialize the quantum state of the cube with superposition"""
        state_dim = 2 ** self.qubit_depth
        state = np.zeros(state_dim, dtype=np.complex128)
        state[0] = 1.0  # Start in ground state
        
        # Apply Hadamard to create superposition
        for i in range(self.qubit_depth):
            state = self._apply_hadamard(state, i)
            
        return state
    
    def _apply_hadamard(self, state: np.ndarray, target_qubit: int) -> np.ndarray:
        """Apply Hadamard gate to target qubit"""
        n = len(state)
        result = np.zeros_like(state)
        h_factor = 1.0 / np.sqrt(2)
        
        for i in range(n):
            # Check if target qubit is 0
            if (i & (1 << target_qubit)) == 0:
                # Target is 0, apply |0⟩ → |+⟩ transformation
                zero_state = i
                one_state = i | (1 << target_qubit)
                result[zero_state] += state[i] * h_factor
                result[one_state] += state[i] * h_factor
            else:
                # Target is 1, apply |1⟩ → |−⟩ transformation
                zero_state = i & ~(1 << target_qubit)
                one_state = i
                result[zero_state] += state[i] * h_factor
                result[one_state] -= state[i] * h_factor
                
        return result
    
    def add_node(self, position: np.ndarray, properties: Dict[str, Any]) -> str:
        """Add a node to the cube at specific position"""
        # Generate unique node ID
        node_id = hashlib.md5(f"{time.time()}:{position}:{properties}".encode()).hexdigest()[:12]
        
        # Adapt position to grid
        grid_position = self._continuous_to_grid(position)
        
        # Initialize node in graph with quantum properties
        self.node_graph.add_node(
            node_id, 
            position=position,
            grid_position=grid_position,
            energy=properties.get('energy', 0.5),
            stability=properties.get('stability', 0.8),
            quantum_phase=properties.get('phase', 0.0),
            properties=properties
        )
        
        # Update energy grid at node position
        self._update_grid_at_position(grid_position, properties.get('energy', 0.5))
        
        # Update quantum phase grid
        phase = properties.get('phase', 0.0)
        self.quantum_phase_grid[grid_position] = np.exp(1j * phase)
        
        return node_id
    
    def _continuous_to_grid(self, position: np.ndarray) -> Tuple:
        """Convert continuous position [-1,1] to grid coordinates"""
        grid_coords = []
        for i, pos in enumerate(position[:self.dimensions]):
            # Map from [-1,1] to [0,resolution-1]
            grid_coord = int((pos + 1) / 2 * (self.resolution - 1))
            grid_coord = max(0, min(self.resolution - 1, grid_coord))
            grid_coords.append(grid_coord)
            
        # Pad with zeros if position has fewer dimensions than the cube
        while len(grid_coords) < self.dimensions:
            grid_coords.append(0)
            
        return tuple(grid_coords)
    
    def _update_grid_at_position(self, grid_pos: Tuple, energy: float):
        """Update energy grid at specified position with energy diffusion"""
        # Apply energy at exact grid position
        self.energy_grid[grid_pos] += energy
        
        # Apply energy diffusion to neighboring grid points (3D differential heat equation)
        diffusion_radius = 2
        diffusion_factor = 0.25
        
        for offset in self._generate_neighborhood(diffusion_radius):
            neighbor_pos = tuple(max(0, min(self.resolution - 1, g + o)) 
                                 for g, o in zip(grid_pos, offset))
            
            if neighbor_pos != grid_pos:
                # Calculate diffusion based on distance
                distance = np.sqrt(sum((a-b)**2 for a, b in zip(grid_pos, neighbor_pos)))
                if distance <= diffusion_radius:
                    diffusion_amount = energy * diffusion_factor * (1 - distance / diffusion_radius)
                    self.energy_grid[neighbor_pos] += diffusion_amount
    
    def _generate_neighborhood(self, radius: int) -> List[Tuple]:
        """Generate neighborhood coordinates within given radius"""
        if self.dimensions == 3:
            # Optimized for 3D case
            offsets = []
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    for z in range(-radius, radius + 1):
                        offsets.append((x, y, z))
            return offsets
        else:
            # General case for any dimension
            # This uses recursive generation of the neighborhood
            return self._recursive_neighborhood([], 0, radius)
    
    def _recursive_neighborhood(self, current: List[int], dim: int, radius: int) -> List[Tuple]:
        if dim == self.dimensions:
            return [tuple(current)]
        
        result = []
        for offset in range(-radius, radius + 1):
            current.append(offset)
            result.extend(self._recursive_neighborhood(current.copy(), dim + 1, radius))
            current.pop()
            
        return result
    
    def connect_nodes(self, node1_id: str, node2_id: str, tension: float = None) -> bool:
        """Connect two nodes with a quantum string having specified tension"""
        if node1_id not in self.node_graph or node2_id not in self.node_graph:
            return False
        
        # Calculate tension if not provided based on quantum distance
        if tension is None:
            pos1 = self.node_graph.nodes[node1_id]['position']
            pos2 = self.node_graph.nodes[node2_id]['position']
            
            # Euclidean distance
            spatial_distance = np.linalg.norm(pos1 - pos2)
            
            # Quantum phase difference - measure of entanglement
            phase1 = self.node_graph.nodes[node1_id].get('quantum_phase', 0)
            phase2 = self.node_graph.nodes[node2_id].get('quantum_phase', 0)
            phase_difference = abs(np.exp(1j * phase1) - np.exp(1j * phase2))
            
            # Calculate tension based on combination of spatial and quantum factors
            tension = (1.0 / (1.0 + spatial_distance)) * (1.0 - phase_difference / 2.0)
            
        # Add edge to graph
        self.node_graph.add_edge(node1_id, node2_id, tension=tension)
        edge_key = tuple(sorted([node1_id, node2_id]))
        self.edge_tensions[edge_key] = tension
        
        # Update tensor fields to reflect this connection
        self._update_tension_fields(node1_id, node2_id, tension)
        
        return True
    
    def _update_tension_fields(self, node1_id: str, node2_id: str, tension: float):
        """Update tensor fields to reflect string tension between nodes"""
        pos1 = self.node_graph.nodes[node1_id]['grid_position']
        pos2 = self.node_graph.nodes[node2_id]['grid_position']
        
        # Generate points along the connection path
        path_points = self._generate_path_points(pos1, pos2, steps=max(self.resolution // 4, 5))
        
        # Update tensor at each point along the path
        for point in path_points:
            if all(0 <= p < self.resolution for p in point):
                # Create rank-2 tensor representing the string tension
                tension_tensor = self._calculate_tension_tensor(pos1, pos2, point, tension)
                
                # Update the tensor field at this point
                self.tensor_fields[point] += tension_tensor
    
    def _generate_path_points(self, start: Tuple, end: Tuple, steps: int) -> List[Tuple]:
        """Generate points along path from start to end using linear interpolation"""
        points = []
        
        for step in range(steps + 1):
            t = step / steps
            point = tuple(int(s + t * (e - s)) for s, e in zip(start, end))
            points.append(point)
            
        return points
    
    def _calculate_tension_tensor(self, pos1: Tuple, pos2: Tuple, point: Tuple, tension: float) -> np.ndarray:
        """Calculate the tension tensor at a point between two connected nodes"""
        # Vector from point to pos1 and pos2
        v1 = np.array(pos1) - np.array(point)
        v2 = np.array(pos2) - np.array(point)
        
        # Normalize vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 0 and norm2 > 0:
            v1 = v1 / norm1
            v2 = v2 / norm2
            
            # Calculate outer product to form rank-2 tensor
            tensor = tension * np.outer(v1[:self.dimensions], v2[:self.dimensions])
            
            # Symmetrize tensor
            tensor = (tensor + tensor.T) / 2
            
            return tensor
        else:
            # Return zero tensor if vectors cannot be normalized
            return np.zeros((self.dimensions, self.dimensions))
    
    def evolve_quantum_state(self, steps: int = 1):
        """Evolve the quantum state based on current tensor fields"""
        for _ in range(steps):
            # Construct Hamiltonian from tensor fields
            hamiltonian = self._construct_hamiltonian()
            
            # Apply quantum evolution
            self.state_vector = self._apply_quantum_evolution(hamiltonian)
            
            # Update quantum phase grid based on new state
            self._update_phase_grid()
    
    def _construct_hamiltonian(self) -> np.ndarray:
        """Construct Hamiltonian matrix from current tensor fields"""
        # Get scalar value representing overall tension
        overall_tension = np.mean(np.abs(self.tensor_fields))
        
        # For computational efficiency, construct a sparse representation
        # focusing on relevant basis states
        state_dim = 2 ** self.qubit_depth
        hamiltonian = np.zeros((state_dim, state_dim), dtype=np.complex128)
        
        # Use energy grid and tensor field to influence the Hamiltonian
        for i in range(state_dim):
            # Diagonal elements - influenced by energy
            binary_representation = format(i, f'0{self.qubit_depth}b')
            energy_factor = sum(int(bit) for bit in binary_representation) / self.qubit_depth
            hamiltonian[i, i] = energy_factor * overall_tension
            
            # Off-diagonal elements - create entanglement based on tensor field
            for j in range(i + 1, state_dim):
                # Only connect states that differ by 1 or 2 bit flips
                bit_diff = bin(i ^ j).count('1')
                if bit_diff <= 2:
                    # Connection strength related to overall tension and bit difference
                    strength = overall_tension * (1.0 / bit_diff) * self.entanglement_factor
                    hamiltonian[i, j] = strength
                    hamiltonian[j, i] = strength
        
        return hamiltonian
    
    def _apply_quantum_evolution(self, hamiltonian: np.ndarray) -> np.ndarray:
        """Apply quantum evolution using the given Hamiltonian"""
        # Compute time-evolution operator using matrix exponential
        # U = exp(-i * H * dt)
        dt = 0.1  # Time step (can be adjusted)
        
        # For better performance, you could use scipy's expm:
        # from scipy.linalg import expm
        # U = expm(-1j * hamiltonian * dt)
        
        # Simplified version using eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        diagonal_exp = np.exp(-1j * eigenvalues * dt)
        U = eigenvectors @ np.diag(diagonal_exp) @ eigenvectors.T.conj()
        
        # Apply evolution operator
        evolved_state = U @ self.state_vector
        
        # Normalize
        norm = np.linalg.norm(evolved_state)
        if norm > 0:
            evolved_state = evolved_state / norm
            
        return evolved_state
    
    def _update_phase_grid(self):
        """Update quantum phase grid based on current quantum state"""
        # Calculate local phase values from quantum state
        # This is a simplified approach - in a full implementation,
        # you would map regions of the grid to specific quantum states
        
        # For each grid point, assign a phase based on local state contribution
        for index in np.ndindex(*([self.resolution] * self.dimensions)):
            # Calculate position in normalized coordinates
            norm_pos = tuple(idx / (self.resolution - 1) for idx in index)
            
            # Map position to quantum state indices
            state_indices = self._map_position_to_states(norm_pos)
            
            # Calculate phase from these states
            phase = 0
            for state_idx, weight in state_indices:
                if state_idx < len(self.state_vector):
                    amplitude = self.state_vector[state_idx]
                    if abs(amplitude) > 0:
                        phase += weight * np.angle(amplitude)
            
            # Update quantum phase grid
            self.quantum_phase_grid[index] = np.exp(1j * phase)
    
    def _map_position_to_states(self, norm_pos: Tuple) -> List[Tuple[int, float]]:
        """Map a normalized position to quantum state indices with weights"""
        # Create mapping from position to key quantum states
        state_indices = []
        
        # Focus on a small subset of states to make calculation efficient
        num_states = min(8, 2 ** self.qubit_depth)
        
        for state_idx in range(num_states):
            # Calculate weight based on position and state index
            # This is a simplified mapping - a real implementation would use
            # a more sophisticated scheme
            binary = format(state_idx, f'0{self.dimensions}b')
            weight = 1.0
            for dim, bit in enumerate(binary[:self.dimensions]):
                pos_val = norm_pos[dim]
                bit_val = int(bit)
                # Weight drops as position differs from bit value
                weight *= 1.0 - abs(pos_val - bit_val)
            
            if weight > 0.01:  # Ignore negligible weights
                state_indices.append((state_idx, weight))
        
        return state_indices
    
    def calculate_tension_field(self) -> np.ndarray:
        """Calculate a scalar tension field from the tensor fields"""
        # Compute the Frobenius norm at each grid point
        tension = np.zeros([self.resolution] * self.dimensions)
        
        for index in np.ndindex(*([self.resolution] * self.dimensions)):
            tensor = self.tensor_fields[index]
            # Frobenius norm: sqrt(sum of squared elements)
            tension[index] = np.sqrt(np.sum(tensor * tensor))
        
        return tension
    
    def extract_network_state(self) -> Dict[str, Any]:
        """Extract current state of the node network for visualization/analysis"""
        tension_field = self.calculate_tension_field()
        nodes_data = []
        edges_data = []
        
        # Node data
        for node_id, data in self.node_graph.nodes(data=True):
            grid_pos = data['grid_position']
            tension_at_node = tension_field[grid_pos] if all(p < self.resolution for p in grid_pos) else 0
            
            node_info = {
                'id': node_id,
                'position': data['position'].tolist() if isinstance(data['position'], np.ndarray) else data['position'],
                'energy': data.get('energy', 0),
                'stability': data.get('stability', 0),
                'local_tension': float(tension_at_node),
                'properties': data.get('properties', {})
            }
            nodes_data.append(node_info)
        
        # Edge data
        for node1, node2, data in self.node_graph.edges(data=True):
            edge_info = {
                'source': node1,
                'target': node2,
                'tension': data.get('tension', 0)
            }
            edges_data.append(edge_info)
        
        # Extract high-tension points for visualization
        high_tension_points = []
        threshold = 0.5 * np.max(tension_field)
        
        # Sample points to avoid overwhelming visualization
        step = max(1, self.resolution // 10)
        for index in np.ndindex(*([slice(0, self.resolution, step)] * self.dimensions)):
            tension = tension_field[index]
            if tension > threshold:
                # Convert grid coordinates to normalized position
                position = tuple(idx / (self.resolution - 1) * 2 - 1 for idx in index)
                high_tension_points.append({
                    'position': position,
                    'tension': float(tension)
                })
        
        return {
            'nodes': nodes_data,
            'edges': edges_data,
            'high_tension_points': high_tension_points,
            'energy_stats': {
                'min': float(np.min(self.energy_grid)),
                'max': float(np.max(self.energy_grid)),
                'mean': float(np.mean(self.energy_grid)),
                'std': float(np.std(self.energy_grid))
            },
            'tension_stats': {
                'min': float(np.min(tension_field)),
                'max': float(np.max(tension_field)),
                'mean': float(np.mean(tension_field)),
                'std': float(np.std(tension_field))
            }
        }
    
    def simulate_step(self):
        """Run a single simulation step, updating all fields and states"""
        # Apply damping to tensor fields
        self.tensor_fields *= self.harmonic_damping
        
        # Update node positions and properties based on current fields
        self._update_nodes()
        
        # Recalculate tension fields based on updated node positions
        self._recalculate_tension_fields()
        
        # Evolve quantum state
        self.evolve_quantum_state(steps=1)
        
        # Apply quantum effects back to classical properties
        self._apply_quantum_to_classical()
    
    def _update_nodes(self):
        """Update node positions based on current tension field"""
        tension_field = self.calculate_tension_field()
        
        for node_id, data in list(self.node_graph.nodes(data=True)):
            grid_pos = data['grid_position']
            
            # Calculate force vector from tension gradient
            force = np.zeros(self.dimensions)
            
            # Calculate gradient using central differences
            for dim in range(self.dimensions):
                # Create offset positions for central difference
                pos_forward = list(grid_pos)
                pos_backward = list(grid_pos)
                
                if grid_pos[dim] + 1 < self.resolution:
                    pos_forward[dim] += 1
                if grid_pos[dim] - 1 >= 0:
                    pos_backward[dim] -= 1
                
                # Calculate gradient using central difference
                tension_forward = tension_field[tuple(pos_forward)]
                tension_backward = tension_field[tuple(pos_backward)]
                gradient = (tension_forward - tension_backward) / 2.0
                
                # Force is negative gradient (move away from high tension)
                force[dim] = -gradient
            
            # Update position based on force
            position = np.array(data['position'])
            
            # Scale force by elasticity and apply
            position += force * self.string_elasticity
            
            # Ensure position stays within valid range [-1, 1]
            position = np.clip(position, -1.0, 1.0)
            
            # Update node position
            self.node_graph.nodes[node_id]['position'] = position
            self.node_graph.nodes[node_id]['grid_position'] = self._continuous_to_grid(position)
            
            # Update node properties based on local field values
            energy = data.get('energy', 0.5)
            
            # Energy influenced by local tension
            local_tension = tension_field[grid_pos]
            energy_change = 0.05 * (local_tension - 0.5)  # -0.025 to 0.025
            
            # Apply energy change
            new_energy = np.clip(energy + energy_change, 0.1, 1.0)
            self.node_graph.nodes[node_id]['energy'] = new_energy
    
    def _recalculate_tension_fields(self):
        """Recalculate all tension fields based on current node positions"""
        # Reset tensor fields (maintain some damping)
        self.tensor_fields *= 0.3
        
        # Update tensor fields for each edge
        for edge in self.node_graph.edges(data=True):
            node1_id, node2_id = edge[0], edge[1]
            tension = edge[2].get('tension', 0.5)
            
            # Update based on current positions
            self._update_tension_fields(node1_id, node2_id, tension)
    
    def _apply_quantum_to_classical(self):
        """Apply quantum state effects to classical properties"""
        # Measure quantum properties
        entanglement_entropy = self._calculate_entanglement_entropy()
        coherence = self._calculate_quantum_coherence()
        
        # Apply to nodes
        for node_id, data in self.node_graph.nodes(data=True):
            grid_pos = data['grid_position']
            
            # Get local quantum phase
            local_phase = np.angle(self.quantum_phase_grid[grid_pos])
            
            # Update node quantum phase
            self.node_graph.nodes[node_id]['quantum_phase'] = local_phase
            
            # Update stability based on entanglement and coherence
            stability = data.get('stability', 0.8)
            stability_change = 0.02 * (coherence - 0.5)  # -0.01 to 0.01
            
            # Apply stability change
            new_stability = np.clip(stability + stability_change, 0.2, 0.95)
            self.node_graph.nodes[node_id]['stability'] = new_stability
    
    def _calculate_entanglement_entropy(self) -> float:
        """Calculate von Neumann entropy of the quantum state"""
        # In a full implementation, this would involve calculating
        # the reduced density matrix and its eigenvalues
        # Simplified version using probabilities
        probabilities = np.abs(self.state_vector) ** 2
        entropy = 0
        for p in probabilities:
            if p > 1e-10:  # Avoid log(0)
                entropy -= p * np.log2(p)
        return entropy
    
    def _calculate_quantum_coherence(self) -> float:
        """Calculate quantum coherence (off-diagonal elements of density matrix)"""
        # Simplified measure using amplitude sums
        return np.sum(np.abs(self.state_vector)) / len(self.state_vector)
