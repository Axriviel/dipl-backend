class UserService:
    def __init__(self, repository):
        self.repository =  repository
    
    def getAll(self):
        return self.repository.getAll()
        
