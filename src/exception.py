import sys # sys module is used to fetch exception details like traceback
# from logger import logging
import logging
# Function to create a detailed error message
def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info() # Get the traceback object from the exception info
    file_name=exc_tb.tb_frame.f_code.co_filename# Get the file name where the error occurred
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error)) # Format a detailed error message with file name, line number, and error message

    return error_message

    
#Custom exception class that extends Python's base Exception class
class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):# Constructor takes original error and sys module
        super().__init__(error_message)# Initialize the base Exception class with the error message
        self.error_message=error_message_detail(error_message,error_detail=error_detail)# Generate and store the detailed error message
    
    def __str__(self):#When the exception is converted to string (e.g., print or str()), return this
        return self.error_message# Return the formatted detailed error message
    
# if __name__=="__main__":
#     try:
#         a=1/0
#     except Exception as e:
#         logging.info("Divide by zero")
#         raise CustomException(e,sys)
    

     