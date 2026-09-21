import json
import os
import urllib.request
from dataclasses import dataclass, asdict

@dataclass
class MaterialProperties:
    name: str
    Eg: float        # Bandgap (eV)
    mu_e: float      # Electron mobility (cm^2/Vs)
    mu_h: float      # Hole mobility (cm^2/Vs)
    tau_e: float     # Electron lifetime (s)
    tau_h: float     # Hole lifetime (s)
    W_pair: float    # Pair creation energy (eV)
    density: float   # Density (g/cm^3)
    atomic_composition: dict # {'Element': fraction}
    
    @property
    def mu_tau_e(self):
        return self.mu_e * self.tau_e
        
    @property
    def mu_tau_h(self):
        return self.mu_h * self.tau_h

class MaterialManager:
    def __init__(self, data_file="data/materials.json"):
        import sys
        if getattr(sys, 'frozen', False):
            # Running as compiled (PyInstaller)
            if hasattr(sys, '_MEIPASS'):
                # --onefile mode: resources are in _MEIPASS
                # We need to distinguish between READ and WRITE paths.
                # data/materials.json should be readable from bundle
                # but if we want to save, we can't write back to _MEIPASS (it's temp/readonly-ish)
                
                # For reading default data:
                self.bundled_data_file = os.path.join(sys._MEIPASS, data_file)
                
                # For user data (writable):
                # On Windows: AppData/Local/FermiSimulator
                # On macOS: ~/Library/Application Support/FermiSimulator
                if sys.platform == 'win32':
                    user_dir = os.path.join(os.environ['LOCALAPPDATA'], 'FermiSimulator')
                elif sys.platform == 'darwin':
                    user_dir = os.path.join(os.path.expanduser('~/Library/Application Support'), 'FermiSimulator')
                else:
                    user_dir = os.path.join(os.path.expanduser('~/.local/share'), 'FermiSimulator')
                
                os.makedirs(user_dir, exist_ok=True)
                self.user_data_file = os.path.join(user_dir, "materials.json")
                
                # If user file doesn't exist, we will use bundled file
                # But self.data_file needs to point to the user file for saving
                self.data_file = self.user_data_file
                
                # Initial load logic needs to be smart (check user first, then bundled)
                self.materials = {}
                self._load_with_fallback()
                return # Skip default load
                
            else:
                # --onedir mode
                base_path = os.path.dirname(sys.executable)
        else:
            # Running from source
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        self.data_file = os.path.join(base_path, data_file)
        self.materials = {}
        self.load_materials()

    def _load_with_fallback(self):
        # 1. Try loading from user writable location
        if os.path.exists(self.user_data_file):
            try:
                with open(self.user_data_file, 'r') as f:
                    data = json.load(f)
                    for name, props in data.items():
                        self.materials[name] = MaterialProperties(**props)
                return
            except Exception as e:
                print(f"Error loading user materials: {e}")
        
        # 2. Fallback to bundled data
        if os.path.exists(self.bundled_data_file):
            try:
                with open(self.bundled_data_file, 'r') as f:
                    data = json.load(f)
                    for name, props in data.items():
                        self.materials[name] = MaterialProperties(**props)
                # Auto-copy to user dir so they have a writable copy immediately?
                # Optional, but good for UX. Let's do it on next save.
            except Exception as e:
                print(f"Error loading bundled materials: {e}")

    def load_materials(self):
        if not os.path.exists(self.data_file):
            print(f"Warning: Material database {self.data_file} not found.")
            return

        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                for name, props in data.items():
                    self.materials[name] = MaterialProperties(**props)
        except Exception as e:
            print(f"Error loading materials: {e}")

    def save_materials(self):
        data = {name: asdict(props) for name, props in self.materials.items()}
        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving materials: {e}")

    def get_material(self, name):
        return self.materials.get(name)

    def get_all_names(self):
        return list(self.materials.keys())

    def fetch_catalog(self, url):
        """
        Fetch material catalog from URL but DO NOT save it yet.
        Returns a dict of available materials.
        """
        try:
            # Handle local file URL simulation
            if url.startswith("file://"):
                local_path = url.replace("file://", "")
                with open(local_path, 'r') as f:
                    data = json.load(f)
            else:
                # Real network request
                with urllib.request.urlopen(url) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode())
                    else:
                        return None, f"HTTP Error: {response.status}"
            
            valid_materials = {}
            for name, props in data.items():
                required_fields = ["name", "Eg", "mu_e", "mu_h", "tau_e", "tau_h", "W_pair", "density", "atomic_composition"]
                if all(k in props for k in required_fields):
                    valid_materials[name] = MaterialProperties(**props)
            
            return valid_materials, "Success"
            
        except Exception as e:
            return None, f"Fetch failed: {str(e)}"

    def import_materials(self, materials_dict):
        """
        Merge a dictionary of MaterialProperties into the local database and save.
        """
        count = 0
        for name, props in materials_dict.items():
            self.materials[name] = props
            count += 1
        
        if count > 0:
            self.save_materials()
        return count

    def delete_material(self, name):
        """
        Delete a material from the local database and save.
        """
        if name in self.materials:
            del self.materials[name]
            self.save_materials()
            return True
        return False

    def update_from_url(self, url):
        """
        Fetch material data from a URL (JSON format) and merge into local database.
        """
        try:
            with urllib.request.urlopen(url) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    count = 0
                    for name, props in data.items():
                        # Validate basic fields exist before adding
                        required_fields = ["name", "Eg", "mu_e", "mu_h", "tau_e", "tau_h", "W_pair", "density", "atomic_composition"]
                        if all(k in props for k in required_fields):
                            self.materials[name] = MaterialProperties(**props)
                            count += 1
                    
                    if count > 0:
                        self.save_materials()
                        return True, f"Updated {count} materials."
                    else:
                        return False, "No valid material data found in response."
                else:
                    return False, f"HTTP Error: {response.status}"
        except Exception as e:
            return False, f"Update failed: {str(e)}"

# Global instance for backward compatibility if needed, 
# but preferably instantiate MaterialManager where needed.
# For now, we load it once to populate the legacy MATERIALS dict if other modules use it directly.
_manager = MaterialManager()
MATERIALS = _manager.materials
