
"""Sample Python module for testing."""

def calculate_fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number.
    
    Args:
        n: The position in the Fibonacci sequence
        
    Returns:
        The nth Fibonacci number
    """
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

def main():
    """Main function demonstrating the calculator."""
    calc = Calculator()
    print(calc.add(5, 3))
    print(calc.multiply(4, 7))
    print(f"Fibonacci(10) = {calculate_fibonacci(10)}")

if __name__ == "__main__":
    main()
