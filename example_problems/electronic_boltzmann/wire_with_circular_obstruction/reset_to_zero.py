import os

start_index = 5000
end_index   = 8000


for index in range(end_index - start_index):
    os.system('mv dump_%06d dump_%06d'%(index+start_index, index))
