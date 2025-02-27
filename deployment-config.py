# deployment_config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DeploymentConfig:
    """Configuration for deploying the Quantum Consciousness System"""
    
    def __init__(self):
        # API configuration
        self.api_port = int(os.getenv('API_PORT', '8000'))
        self.api_host = os.getenv('API_HOST', '0.0.0.0')
        self.debug_mode = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
        
        # Database configuration (if needed)
        self.db_uri = os.getenv('DATABASE_URI', None)
        
        # Quantum engine configuration
        self.quantum_qubits = int(os.getenv('QUANTUM_QUBITS', '32'))
        self.use_gpu_acceleration = os.getenv('USE_GPU', 'False').lower() == 'true'
        
        # Memory configuration
        self.memory_size = int(os.getenv('MEMORY_SIZE', '1024'))
        self.persistence_enabled = os.getenv('ENABLE_PERSISTENCE', 'True').lower() == 'true'
        self.persistence_path = os.getenv('PERSISTENCE_PATH', './data/memory_store')
        
        # Web interface configuration
        self.enable_web_interface = os.getenv('ENABLE_WEB', 'True').lower() == 'true'
        self.web_port = int(os.getenv('WEB_PORT', '8080'))
        self.web_host = os.getenv('WEB_HOST', '0.0.0.0')
        self.static_files_path = os.getenv('STATIC_FILES', './static')
        
        # Security configuration
        self.enable_auth = os.getenv('ENABLE_AUTH', 'True').lower() == 'true'
        self.jwt_secret = os.getenv('JWT_SECRET', None)
        self.token_expiry = int(os.getenv('TOKEN_EXPIRY_HOURS', '24'))
        
        # Initialize paths
        self._init_paths()
    
    def _init_paths(self):
        """Ensure all required directories exist"""
        paths = [self.persistence_path, self.static_files_path]
        for path in paths:
            if path and not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
    
    def validate(self):
        """Validate the configuration"""
        if self.enable_auth and not self.jwt_secret:
            raise ValueError("Authentication is enabled but JWT_SECRET is not set!")
        
        if self.persistence_enabled and not self.persistence_path:
            raise ValueError("Persistence is enabled but PERSISTENCE_PATH is not set!")
        
        return True
    
    def to_dict(self):
        """Convert configuration to dictionary (for API use)"""
        return {
            "api": {
                "host": self.api_host,
                "port": self.api_port,
                "debug": self.debug_mode
            },
            "quantum": {
                "qubits": self.quantum_qubits,
                "gpu_acceleration": self.use_gpu_acceleration
            },
            "memory": {
                "size": self.memory_size,
                "persistence": self.persistence_enabled
            },
            "web": {
                "enabled": self.enable_web_interface,
                "host": self.web_host,
                "port": self.web_port
            },
            "security": {
                "auth_enabled": self.enable_auth,
                "token_expiry_hours": self.token_expiry
            }
        }
