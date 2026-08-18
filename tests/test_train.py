import pytest
from src.config import config, ModelConfig, PathConfig

def test_config_defaults():
    assert config.model.random_state == 42
    assert config.model.test_size == 0.2
    assert config.paths.data_dir == "data"
    
def test_pydantic_validation():
    with pytest.raises(ValueError):
        ModelConfig(test_size=1.5)  # Should raise error because test_size <= 1.0
        
    with pytest.raises(ValueError):
        ModelConfig(rf_n_estimators=-10)  # Should raise error because gt=0
