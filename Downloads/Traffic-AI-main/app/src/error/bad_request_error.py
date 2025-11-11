class BadRequestError(Exception):
    message: str
    code: int = 400

    def __init__(self, message="Bad Request"):
        self.message = message
        super().__init__(self.message)