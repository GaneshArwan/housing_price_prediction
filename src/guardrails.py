from pydantic import BaseModel, Field

class PredictionGuardrail(BaseModel):
    """
    Guardrail to ensure predicted home prices are realistic and structurally valid.
    """
    predicted_price: float = Field(..., description="The predicted sale price of the house.", gt=0)
    
    @classmethod
    def validate_prediction(cls, price: float) -> bool:
        try:
            cls(predicted_price=price)
            # Add arbitrary business logic guardrails
            if price > 10000000 or price < 1000:
                print(f"Warning: Price {price} failed extreme value guardrails.")
                return False
            return True
        except ValueError as e:
            print(f"Guardrail validation failed: {e}")
            return False
