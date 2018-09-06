from compare_reg import reg
from compare_parallel import par
import matplotlib.pyplot as plt
import pandas as pd

filenames = ['hypertextdata','haggledata','reality_miningdata']

times_reg = []
times_par = []
nms = ['Hypertext','Haggle','Reality\nMining']
for fname in filenames:
	r = reg(fname)
	p = par(fname)
	
	times_reg.append(r)
	times_par.append(p)
	
	print('done with', fname)
	
	
d = {'':nms,'acc':times_reg,'timer':times_par}
df = pd.DataFrame(data=d)
df = df.set_index('')

fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()

width = .4
a = df.acc.plot.bar(color='black',ax=ax, width=width,position=1,label='Unparallelized',rot=0)
b = df.timer.plot(kind='bar',color='slategrey', ax=ax2,width=width,position=0,label='Parallelized')

ax.set_ylabel('Time (s)')
ax2.set_ylabel('Time (s)')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title('Comparing Parallelized Computation Time')
plt.xlabel('')
plt.xlim(-.5,4)
plt.savefig('test_parallel_results.png')