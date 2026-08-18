from pydantic import BaseModel, Field
import os

class ModelConfig(BaseModel):
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    test_size: float = Field(default=0.2, description="Test split ratio", ge=0.0, le=1.0)
    rf_n_estimators: int = Field(default=100, description="Number of trees in Random Forest", gt=0)
    rf_max_depth: int | None = Field(default=None, description="Maximum depth of the trees")

class PathConfig(BaseModel):
    data_dir: str = Field(default="data", description="Directory to store dataset")
    reports_dir: str = Field(default="reports/figures", description="Directory to store figures and plots")
    model_dir: str = Field(default="models", description="Directory to store saved models")

class AppConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    paths: PathConfig = PathConfig()

    def create_dirs(self):
        """Ensure necessary directories exist."""
        os.makedirs(self.paths.data_dir, exist_ok=True)
        os.makedirs(self.paths.reports_dir, exist_ok=True)
        os.makedirs(self.paths.model_dir, exist_ok=True)

# Global configuration instance
config = AppConfig()
config.create_dirs()
