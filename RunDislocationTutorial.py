import sys,os,glob


Generate=sys.argv[1]
dir0=os.getcwd()
print(dir0)
if Generate=='T':

    os.system('rm *.png')

    if os.path.exists('datasets'):
        os.system('rm -rf datasets')

    os.system('mkdir datasets')
    
    print('Dislocation Data will be generated from scratch...')

    os.system('python GenerateDislocationData.py Nucleation')

    os.system('python GenerateDislocationData.py Glide')

print('RunSEAmodes about to run...')

os.system('python runSEAmodes.py Nucleation')

os.system('python runSEAmodes.py Glide')
