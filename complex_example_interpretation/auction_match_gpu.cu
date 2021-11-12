#include <cstdio>

// b: batch size; n: points num;  xyz1,xyz2: two tensor,(b,n,3); matchl,matchr: two tensors,(b,n); cost,tensor,(b,n,n)
// __restrict__保证不存在别名，那么在修改其中一个指针指向的内存中的值并不会影响另一个指正指向的内存
__global__ void AuctionMatchKernel(int b,int n,const float * __restrict__ xyz1,const float * __restrict__ xyz2,int * matchl,int * matchr,float * cost){
	//this kernel handles up to 4096 points
	const int NMax=4096;  // 最多可以处理的点数
    // __shared__ 块内共享： https://www.cxyzjd.com/article/dcrmg/54880471
	__shared__ short Queue[NMax];
	__shared__ short matchrbuf[NMax];
	__shared__ float pricer[NMax];
	__shared__ float bests[32][3];
	__shared__ int qhead,qlen;
	const int BufLen=2048;
	__shared__ float buf[BufLen];  // buf的用意是什么？
	/**
	    我们需要计算一个点云中所有点到另一个点云所有点的距离，总共n*n个数，
	    最简单的方式是我用一个模块中的512个线程每次处理点云1中的一个点到点云2中的n个点的距离，每个线程处理n/512个数，
	    这是一次循环，总共循环n次，每次循环，都要等所有线程的运算都结束了以后，才能进行下一次循环，只要还有线程忙碌，提前结束的线程都需要等待。
	    为了缓解这里的等待时间，可以一次处理点云1中的k个点到点云2中的所有点的距离，这样就减少了k-1次等待时间

	**/

    // 所有运算在这个for循环中完成
	for (int bno=blockIdx.x;bno<b;bno+=gridDim.x){  // gridDim.x为32，blockIdx.x为当前线程所在的块
	    // 这里其实就是将batch分组，保证每次处理的patch数量不多于block的数量，方便后面算法实现
	    /**
            这里bno从当前blockIdx开始，每次循环加上grid的宽度，加入bno一开始是2，那么下一次就是2+32=34，如果batch_size也是32，那么循环结束
            如果batch_size小于32，那么batch_size之后的block相当于是闲置的。
            如果batch_size是64，gridDim等于32，那么就要所有线程分两次处理完一个batch。
            也就是说，将batch_size按照一个网格中所有的块进行分组，依次处理batch中的数据
            所以如果gridDim等于32，那么batch_size最好是32的整数倍数，这样就可以使得块中的线程因为无法整除而闲置
	    **/


		int cnt=0;
		float tolerance=1e-4;

        // 初始化矩阵，对于一个n*p的矩阵，每个block的512个线程处理矩阵的一行，现将矩阵行优先地排开
        // 对于一个线程，其处理的数据在所有数据的bno*n ~ bno*(n+1)之间的这些数据
        // 将一行n个数据按照一个块中的线程分组，每个线程需要处理组数个数据（假设整除）
        // threadIdx.x表示线程在块内的序号，其应该处理的那些数据应该是bno*n+j，bno*n+j+blockDim.x，bno*n+j+blockDim.x*2...
        // 所以下面的一个for循环可以对任意形状的矩阵进行赋值
		for (int j=threadIdx.x;j<n;j+=blockDim.x)
			matchl[bno*n+j]=-1;  // matchl矩阵, (b,n), 用-1填充
		for (int j=threadIdx.x;j<n;j+=blockDim.x)
			matchrbuf[j]=-1;  // matchrbuf矩阵, (n=4096,1), 用-1填充
		for (int j=threadIdx.x;j<n;j+=blockDim.x)
			Queue[j]=j;  // Queue矩阵，(n=4096,1)，使用0~4095填充
		for (int j=threadIdx.x;j<n;j+=blockDim.x)
			pricer[j]=0;  // pricer矩阵，(n=4096,1)，使用0填充

		const int Block=512;  // 为什么要512个点一批进行处理，为什么不直接计算xyz1中所有点到xyz2所有点的距离呢？

        // 对于b*n*3的点云，第bno个block内的线程处理第bno个patch (如果batch_size大于block的数量，那么后面的patch无法被处理到？由k0控制)
		for (int k0=0;k0<n;k0+=Block){  // k0是当前要处理的512个点（或小于）的起始index，k1终止位置
			int k1=min(n,k0+Block);  // 如果n无法被Block整除，那么每一轮k1-k0都是512，否则最后一轮会是小于512的余数
			for (int k=threadIdx.x;k<(k1-k0)*3;k+=blockDim.x)  // threadIdx.x from 0 to 511, k from 0 to 512*3
				buf[k]=xyz1[bno*n*3+k0*3+k];  // buf，(2048,1); xyz1,(b,n,3)
				// bno*n*3保证第bno个block内的线程只处理点云的第bno行，每块处理一个patch
				// k0每轮增加512，k0*3保证每一轮for循环（外层），可以处理512个点，k1=min(n,k0+Block)可以保证不会访问超多n个点

            /**
                所以这里对于buf赋值的整体思路是：
                    对于一个(b,n,3)的张量，我们每次用buf存储512个点的数据，实际上只需要512*3=1536个数字，
                    这里申请2048个数字可能是为了保证一定的余量，其实不用这么大。
                    首先将batch分组，由于buf每次保存512个点，所以需要将b*n*3个点按照每组512个点进行分组
                    所以，首先我们需要使用`for (int k0=0;k0<n;k0+=Block)`，Block=512保证每次都会处理512个点，
                    比如第三次循环，k0 = 0+512+512=1024，说明前1024个点已经被处理，也就是处理了k0*3个数字了。
                    当已经处理了k0*3个数字后，接下来要继续处理512个点，但是剩下的点的数量不一定有512个，所以使用k1=min(n,k0+Block)
                    保证我们不会访问越界。
                    对于512个点，需要继续分组给grid中的block的线程进行处理，外层循环k0保证遍历了一个点云的所有点，
                    但是我们并不是只有一个patch，所以这里是用每个block处理一个patch，
                    所以对于一个线程，其处于第bno%32个block的threadIdx.x位置上，应该处理的是第bno%32个patch的512*3个数字中的
                    第threadIdx.x、threadIdx.x+blockDim.x、threadIdx.x+blockDim.x+blockDim.x……个数字，
                    如果bno小于batch_size，那么后面的patch怎么被处理到呢？实际上在最开始的for循环已经将batch进行了分组，
                    所以如果batch_size大于block_num，那么每次只会处理不多于block_num个patch,
            **/
            // 一个grid中的所有块是逐个执行的吗？如果不是的话，每个块处理一个patch，每个patch都会取512个点，那么buf中的数据会冲突啊，
            // 我们应该保证buf中的数据来自于同一个patch的512个点。
            // 注意，buf向量是__shared__的，也就是说仅仅是块被共享的，并不是全局共享的
			__syncthreads();  // 一个块中的线程在此刻同步，保证buf矩阵都被赋值完成了


            /**
                为什么套路是查看当前线程的idx，这个线程就处理第idx个数据，接下来处理第idx+blockDim.x个数据？
                这样可以保证线程之间不会同时访问同一个元素，所有线程的序号从0开始，blockDim.x-1结束，它们都加上blockDim.x之后，
                序号从blockDim.x开始，2*blockDim.x-1结束可以一块一块地处理数据。
            **/

            /**
                cost矩阵的形状是(b,n,n)，表示b个推理得到的patch中的n个点到对应gt中的patch的的n个点的距离。
                其中（i,j,k）位置的值便是pred得到的第i个patch的第j个点和gt中的第i个patch第k个点的欧式距离。
                所以对于一个线程，其块号为bno，块内线程序号为j，那么我们在上一个循环已经将第bno个patch的512个点存储到了buf[]
                我们用j遍历gt中第bno个patch的所有点，对于每个点我们计算buf中512个点到这个点的距离，
                并存储到cost矩阵中，位置为（bno,k,j），对应的位置为bno*n*n+k*n+j。
                可是这里为什么使用的是cost[blockIdx.x*n*n+k*n+j]，blockIdx.x的大小范围是0~32，如果batch大于这个数了怎么办？
                没关系，因为grid一次只能处理blockIdx.x对点云，总共有b组点云，分成很多组进行处理。
                对于后面的组，前面的组计算的cost矩阵完全没用，直接覆盖就好了。
                所以cost矩阵是过大了的，其实仅仅需要blockIdx.x*n*n就行了，但是cost设置为b*n*n非常保险，
                如果blockIdx.x大于b，则后面的block是闲置的，仅前b个会忙碌，不会访问溢出。
            **/
			for (int j=threadIdx.x;j<n;j+=blockDim.x){  // j from 0 to 4096，处理一个patch中所有的点，一次循环里对坐标的三个数字操作
			    // bno*n*3表示依然是每个block处理一个patch，j*3表示第j个点的起始位置
				float x2=xyz2[bno*n*3+j*3+0];
				float y2=xyz2[bno*n*3+j*3+1];
				float z2=xyz2[bno*n*3+j*3+2];
				for (int k=k0;k<k1;k++){  // k0是当前要处理的512个点（或小于）在预测的patch的起始index，k1为终止位置
					float x1=buf[(k-k0)*3+0];
					float y1=buf[(k-k0)*3+1];
					float z1=buf[(k-k0)*3+2];
					float d=sqrtf((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2));
					cost[blockIdx.x*n*n+k*n+j]=d;
				}
			}
			__syncthreads();  // 512个点的距离到gt中所有点的距离全部计算好了
		}
		// 这个for循环结束以后，cost矩阵就计算好了

		//calculate the distacne
		if (threadIdx.x==0){
		// 如果不是第一个线程，这两个变量岂不是无法初始化
		// 并不是，这两个变量是__shared__的，会被块内所有线程共享，这里仅在第一个线程里初始化该变量，是防止重复初始化，
		// 后面再加一个同步指令就可以保证块中所有线程在同步时刻这两个变量都被初始化了，并且只被初始化了一次
		// 观察到__shared__变量在赋值后通常都要同步一次，保证块内线程对该共享变量的操作都已经结束，
		// 否则一些先执行的线程继续走下面的逻辑，但是使用的共享变量一边被后面的访问，一遍被前面的修改，造成错误
			qhead=0;
			qlen=n;
		}
		__syncthreads();  // 等待共享变量初始化完成


		int loaded=0; // 寄存器变量，线程独享
		float value9,value10,value11,value12,value13,value14,value15,value16; // 寄存器变量，线程独享
		while (qlen){  // n个点执行n次while循环
			int i=Queue[qhead];  // 当前循环要处理xyz1中的第i个点 // 寄存器变量，线程独享
			int i2; // 寄存器变量，线程独享
			if (qhead+1<n)  // 如果i不是最后一个点的序号，则i2是i后面那个点的序号，如果i已经是最后一个点的序号了，i2则表示第一个点的序号
				i2=Queue[qhead+1];  // 当前循环要处理xyz1中的第i2个点,处于i后面
			else
				i2=Queue[0];
			float best=1e38f,best2=1e38f; // 寄存器变量，线程独享
			int bestj=0; // 寄存器变量，线程独享
			if (n==blockDim.x*8){  // 如果n是块长度的8倍，4096/512=8，每个线程处理8个点，512个线程一次解决4096个点的运算
				int j=threadIdx.x;
				float value1,value2,value3,value4,value5,value6,value7,value8;
				if (loaded){
				    // pricer，（4096，1），最开始使用0填充
					value1=value9+pricer[j];
					value2=value10+pricer[j+blockDim.x];
					value3=value11+pricer[j+blockDim.x*2];
					value4=value12+pricer[j+blockDim.x*3];
					value5=value13+pricer[j+blockDim.x*4];
					value6=value14+pricer[j+blockDim.x*5];
					value7=value15+pricer[j+blockDim.x*6];
					value8=value16+pricer[j+blockDim.x*7];
					loaded=0;
				}else{
                    /**
                        cost[blockIdx.x*n*n+i*n+j+blockDim.x * t]表示：
                        xyz1中的第blockIdx.x个patch中的第i个点到xyz2中第blockIdx.x个patch中的第（j+blockDim.x * t）个点的距离
                        pricer(4096,1),初始化使用0填充， pricer[j+blockDim.x*t]表示产品的价格，也就是xyz1中的点云的权重
                        总共有32个块，每个块有512个线程，对于一个线程，它首先确定自己所在的块处理的是哪个点云(blockIdx.x*n*n)，
                        然后确定自己要处理xyz1中的那个点(i*n)，
                        value1~8八个值分别表示这个点到达另外八个点的cost，这八个点的位置是这样计算的：
                        首先对于xyz1中的每个点，我们需要考虑它到xyz2中的对应patch的4096个点的cost，
                        我们将这4096个点分成8组，为什么是八组？因为一开始我们就判断了点数4096是blockDim.x的八倍，
                        这样每个线程需要计算八组中相同位置的8个值，记为value1~8
                    **/
					value1=cost[blockIdx.x*n*n+i*n+j]+pricer[j];
					value2=cost[blockIdx.x*n*n+i*n+j+blockDim.x]+pricer[j+blockDim.x];
					value3=cost[blockIdx.x*n*n+i*n+j+blockDim.x*2]+pricer[j+blockDim.x*2];
					value4=cost[blockIdx.x*n*n+i*n+j+blockDim.x*3]+pricer[j+blockDim.x*3];
					value5=cost[blockIdx.x*n*n+i*n+j+blockDim.x*4]+pricer[j+blockDim.x*4];
					value6=cost[blockIdx.x*n*n+i*n+j+blockDim.x*5]+pricer[j+blockDim.x*5];
					value7=cost[blockIdx.x*n*n+i*n+j+blockDim.x*6]+pricer[j+blockDim.x*6];
					value8=cost[blockIdx.x*n*n+i*n+j+blockDim.x*7]+pricer[j+blockDim.x*7];
					/**
                        value9~16是xyz1第blockIdx.x个patch中的第i2个点到xyz2中第blockIdx.x个patch中的的8个点的cost
                        相同块的另一个线程执行时，是第i2个点到另外八个点的cost，
                        一个块内的所有线程可以将xyz1中第i2个点到xyz2中第blockIdx.x个patch中的所有点的cost遍历完
                        value9~16在本轮迭代中并没有使用，而是在下一次迭代中被使用
					**/
					value9=cost[blockIdx.x*n*n+i2*n+j];
					value10=cost[blockIdx.x*n*n+i2*n+j+blockDim.x];
					value11=cost[blockIdx.x*n*n+i2*n+j+blockDim.x*2];
					value12=cost[blockIdx.x*n*n+i2*n+j+blockDim.x*3];
					value13=cost[blockIdx.x*n*n+i2*n+j+blockDim.x*4];
					value14=cost[blockIdx.x*n*n+i2*n+j+blockDim.x*5];
					value15=cost[blockIdx.x*n*n+i2*n+j+blockDim.x*6];
					value16=cost[blockIdx.x*n*n+i2*n+j+blockDim.x*7];

					// 注意：同一个线程对应的value1~8对应的是xyz1中第i个点到xyz2中的八个点的cost，
					// value9~16对应的是xyz1中第i2个点到xyz2中的同样八个点的cost

					loaded=qlen>1;  // qlen用于遍历xyz1中的n个点，每遍历一次会减少，所以一开始loaded会是True，到最后一个点会是False
				}
				int vj,vj2,vj3,vj4;
				if (value1<value2){
					vj=j;  // value1小，取其对应点的index
				}else{
					vj=j+blockDim.x;  // value2小，取其对应点的index
					float t=value1;  // 并且交换value1和value2，保证value1更小
					value1=value2;
					value2=t;
				}
				if (value3<value4){
					vj2=j+blockDim.x*2;
				}else{
					vj2=j+blockDim.x*3;
					float t=value3;
					value3=value4;
					value4=t;
				}
				if (value5<value6){
					vj3=j+blockDim.x*4;
				}else{
					vj3=j+blockDim.x*5;
					float t=value5;
					value5=value6;
					value6=t;
				}
				if (value7<value8){
					vj4=j+blockDim.x*6;
				}else{
					vj4=j+blockDim.x*7;
					float t=value7;
					value7=value8;
					value8=t;
				}
				// 到此为止，将value1~8中两两较小的值保存在奇数位置
				if (value1<value3){
					value2=fminf(value2,value3);  // 1<2，3，4，3<4, 次小的在2和3之间，保存进value2
				}else{
					value2=fminf(value1,value4);  // 3<1，2，4，1<2, 次小的在1和4之间，保存进value2
					value1=value3;
					vj=vj2;
				}
				if (value5<value7){
					value6=fminf(value6,value7);  // 5<6，7，8，7<8, 次小的在6和7之间，保存进value6
				}else{
					value6=fminf(value5,value8);  // 7<5，6，8，5<6, 次小的在5和8之间，保存进value6
					value5=value7;
					vj3=vj4;
				}
				// 到此为止，将value1，3，5，7中两两较小的保存在value1和value5的位置，次小的保存在value2和value6中，下面比较这四个
				if (value1<value5){
					best=value1;
					bestj=vj;
					best2=fminf(value2,value5);  // 1<2，5，6,5<6， 次小的在2和5之间，保存进best2
				}else{
					best2=fminf(value1,value6);  // 5<1，2，6,1<2， 次小的在1和6之间，保存进best2
					best=value5;
					bestj=vj3;
				}
				// 至此，八个数最小的保存在best中，次小的保存在best2中
			}else if (n>=blockDim.x*4){
				for (int j=threadIdx.x;j<n;j+=blockDim.x*4){
					float value1=cost[blockIdx.x*n*n+i*n+j]+pricer[j];
					float value2=cost[blockIdx.x*n*n+i*n+j+blockDim.x]+pricer[j+blockDim.x];
					float value3=cost[blockIdx.x*n*n+i*n+j+blockDim.x*2]+pricer[j+blockDim.x*2];
					float value4=cost[blockIdx.x*n*n+i*n+j+blockDim.x*3]+pricer[j+blockDim.x*3];
					int vj,vj2;
					if (value1<value2){
						vj=j;
					}else{
						vj=j+blockDim.x;
						float t=value1;
						value1=value2;
						value2=t;
					}
					if (value3<value4){
						vj2=j+blockDim.x*2;
					}else{
						vj2=j+blockDim.x*3;
						float t=value3;
						value3=value4;
						value4=t;
					}
					if (value1<value3){
						value2=fminf(value2,value3);
					}else{
						value2=fminf(value1,value4);
						value1=value3;
						vj=vj2;
					}
					if (best<value1){
						best2=fminf(best2,value1);
					}else{
						best2=fminf(best,value2);
						best=value1;
						bestj=vj;
					}
				}
			}else if (n>=blockDim.x*2){
				for (int j=threadIdx.x;j<n;j+=blockDim.x*2){
					float value1=cost[blockIdx.x*n*n+i*n+j]+pricer[j];
					float value2=cost[blockIdx.x*n*n+i*n+j+blockDim.x]+pricer[j+blockDim.x];
					int vj;
					if (value1<value2){
						vj=j;
					}else{
						vj=j+blockDim.x;
						float t=value1;
						value1=value2;
						value2=t;
					}
					if (best<value1){
						best2=fminf(best2,value1);
					}else{
						best2=fminf(best,value2);
						best=value1;
						bestj=vj;
					}
				}
			}else{
			    // 对于xyz1中的第blockIdx.x个patch中的第i个点到xyz2中第blockIdx.x个patch中的所有点依次计算距离，
			    // 使用best和best2记录最大和最小值。
			    // 下面这种for循环没有使用规约，一个块中的线程计算点云1中第i个点到点云2中所有点的最短距离，
			    // 每个线程需要处理n/blockDim.x（+1）个点，每个线程就需要经过n/blockDim.x（+1）次循环，
			    // 这种循环的缺点需要进一步分析 todo
				for (int j=threadIdx.x;j<n;j+=blockDim.x){
					float value=cost[blockIdx.x*n*n+i*n+j]+pricer[j];
					if (best<value){
						best2=fminf(best2,value);
					}else{
						best2=best;
						bestj=j;
						best=value;
					}
				}
			}

			// 这里不同线程负责计算不同部分的最小值，为什么不用同步？

			/**
			    由于上面在计算一个点到4096个点的距离的最小值时，分成了多个线程计算，32个线程都有着自己负责的数据的最小值
			    为了得到一个总的最小值，需要将这些最小值放在一起继续对比，
			    但是这些最小值都存放在各自线程的寄存器中，并不能直接共享，
			    所以这里要用线程束洗牌指令将不同的线程中的数据合并起来
			**/
            for (int i=16;i>0;i>>=1){  // i = 16，8，4，2，1
                // 得到当前线程在warp中的编号（0~32）减去i的线程的 best、best2、bestj 变量的值，分别存在 b1、b2、bj 中
                // 根据之后填充bests的逻辑来看，这个循环可以将一个warp中的所有线程中的最小值进行比较，并最终存放在0，32，64，……，480这些线程上
                // 也就是每个warp的最后一个线程会和warp内的所有线程进行比较，得到这32个线程中的最小值
                float b1=__shfl_down_sync(0xFFFFFFFF,best,i,32);
                float b2=__shfl_down_sync(0xFFFFFFFF,best2,i,32);
                int bj=__shfl_down_sync(0xFFFFFFFF,bestj,i,32);

                // 比较另一个线程中的最小值、次小值和当前线程中的最小值、次小值，
                if (best<b1){  // best<b1<b2, best<best2, 次小值在b1和best2中
                    best2=fminf(b1,best2);
                }else{  // b1<best<best2, b1<b2, 次小值在best和b2中
                    best=b1;
                    best2=fminf(best,b2);
                    bestj=bj;
                }
            }
            if ((threadIdx.x&31)==0){  // 符合条件的是0，32，64，……，480，可以直接用%的
                /**
                    bests的形状是(32, 3)，我们有512个线程，将512个线程的结果分32组存放在bests中
                    线程序号为threadIdx.x的线程结果存放在bests[threadIdx.x % 32][:]的三个位置上
                    注意，并不会冲突，因为符合条件的线程id为0，32，64，……，480，它们分别存储在bests的0，1，2，……，15位置上
                **/
                bests[threadIdx.x>>5][0]=best;
                bests[threadIdx.x>>5][1]=best2;
                *(int*)&bests[threadIdx.x>>5][2]=bestj;  // 这是什么语法
            }
            __syncthreads();  // 保证bests矩阵赋值完成

            int nn=blockDim.x>>5;  // 512>>5=16

            /**
                至此，对于xyz中的任意一个点，我们得到了它到xyz中对应patch的4096个点的距离中最小的16个，分别存在bests的前十六行。
                接下来，很自然，我们要根据这十六个距离得到这个点到4096个点的最近距离。
                根据后面的逻辑，这里会将最优结果保存在第0个线程中
            **/
            if (threadIdx.x<nn){
                best=bests[threadIdx.x][0];
                best2=bests[threadIdx.x][1];
                bestj=*(int*)&bests[threadIdx.x][2];
                for (int i=nn>>1;i>0;i>>=1){  // i = 8,4,2,1
                    float b1=__shfl_down_sync(0xFFFFFFFF,best,i,32);
                    float b2=__shfl_down_sync(0xFFFFFFFF,best2,i,32);
                    int bj=__shfl_down_sync(0xFFFFFFFF,bestj,i,32);
                    if (best<b1){
                        best2=fminf(b1,best2);
                    }else{
                        best=b1;
                        best2=fminf(best,b2);
                        bestj=bj;
                    }
                }
            }

            if (threadIdx.x==0){
                float delta=best2-best+tolerance;  // 这个block负责的xyz1中的这个点到xyz2中的4096个点的最近距离、次近距离
                qhead++;
                qlen--;  // qlen = 4096，4095，……，1
                if (qhead>=n)  // 如果qhead遍历到xyz1对应patch的最后一点，再从第一个点重新遍历
                    qhead-=n;
                int old=matchrbuf[bestj];  // （4096，1），目前谁到第二个点云中第bestj个点最近
                pricer[bestj]+=delta;  // 一个点的价格等于这个点到达最近点和次近点的距离差
                /**
                    为什么要这么定义price？
                    我的理解是，一个点的价格为最近点和次近点的距离差，
                    这个距离越大，说明如果把这个点指派到非最近点会造成的结果更差。
                    为了整体均衡，我们需要指派一些点不和它们的最近点匹配，
                    但是如果能够保证这些点是距离次近点没那么远的点，
                    而那些距离次近点远很多的点，最好不要动它们
                **/


                cnt++;  // while循环的次数
                if (old!=-1){  // matchrbuf是用-1填充的，所以如果之前没有那个点到这个点最近，那么old==1
                    int ql=qlen;
                    int tail=qhead+ql;  // tail =
                    qlen=ql+1;
                    if (tail>=n)
                        tail-=n;
                    Queue[tail]=old;
                }
                if (cnt==(40*n)){
                    if (tolerance==1.0)
                        qlen=0;
                    tolerance=fminf(1.0,tolerance*100);
                    cnt=0;
                }
            }

            __syncthreads();
            if (threadIdx.x==0){  // 赋值操作仅需要一个线程来完成就行，避免多个线程同时写一块内存
                matchrbuf[bestj]=i;  // 第二个点云中bestj个点与第一个点云中第i个点配对
            }
		}
		__syncthreads();  // matchrbuf是__shared__，需要在这里同步块内线程



		for (int j=threadIdx.x;j<n;j+=blockDim.x)
		    // 使用matchrbuf中每个点对应的最近点的序号更新matchr
		    // matchr表示第二个点云中的每个点和第一个点云中每个点的配对信息
			matchr[bno*n+j]=matchrbuf[j];
		for (int j=threadIdx.x;j<n;j+=blockDim.x)
		    // 更新matchl，保证和matchr中的配对情况一致
			matchl[bno*n+matchrbuf[j]]=j;
		__syncthreads();
		// 到此为止，点的配对情况计算完成，分别保存在matchr和matchl中，我们只需要matchl就行
	}
}
void AuctionMatchLauncher(int b,int n,const float * xyz1,const float * xyz2,int * matchl,int * matchr,float * cost){
	AuctionMatchKernel<<<32,512>>>(b,n,xyz1,xyz2,matchl,matchr,cost);
    // 一个grid包含32个block，每个block包含512个线程，均为线性排布
}

