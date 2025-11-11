class NotFoundError(Exception):
    message: str
    code: int = 404
    
    def __init__(self, message="Resource not found"):
        self.message = message
        super().__init__(self.message)