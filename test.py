class Calculator:
    """
    A simple calculator class for basic arithmetic operations.
    """

    def add_numbers(self,a,b):
        """Function to add two numbers."""
        return a+b
    
    def subtract_numbers(self, a, b):
        """Function to subtract two numbers."""
        return a - b
    
    def multiply_numbers(self, a, b):
        """Function to multiply two numbers."""
        return a * b
    
    def divide_numbers(self, a, b):
        """Function to divide two numbers."""
        if b == 0:
            return "Error: Division by zero"
        return a / b
    
# Create an object (instance) of the Calculator class
calc = Calculator()

# Example usage with the object
