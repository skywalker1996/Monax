from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import CPULimitedHost
from mininet.link import TCLink
from mininet.util import dumpNodeConnections,custom,pmonitor
from mininet.log import setLogLevel
import time
from mininet.cli import CLI
from mininet.node import OVSController

### server---client---middleware---database
class SingleSwitchTopo( Topo ):
    "Single switch connected to n hosts."
    def build( self, n=2 ):
        switch = self.addSwitch( 's1' )
        for h in range(n-1):
            # Each host gets 50%/n of system CPU
            host = self.addHost( 'h%s' % (h + 1))
            # 10 Mbps, 5ms delay, 2% loss, 1000 packet queue
            self.addLink( host, switch,bw=500, delay='5ms', loss=0, max_queue_size=409600, use_htb=True )
        ## host for middleware
        host = self.addHost( 'h%s' % (h + 2))
        self.addLink( host, switch, bw=500, delay='5ms', loss=0, max_queue_size=409600, use_htb=True)

        #visualization
        host = self.addHost( 'h%s' % (h + 3))
        self.addLink( host, switch, bw=500, delay='1ms', loss=0, max_queue_size=1024, use_htb=True)

def Testing():
    "Create network and run simple performance test"

    server_num = 1
    client_num = 1
    middleware_num = 1

    host_num = server_num + client_num + middleware_num
    topo = SingleSwitchTopo(n=host_num)
    net = Mininet(topo=topo, link=TCLink)
    net.addNAT().configDefault()
    net.start()
    hosts = {}
    process_list = {}

    for i in range(host_num+1):
        hosts['h%s' % (i + 1)] = net.get('h%s' % (i + 1))
    

    #start the process
    print('start the server/client/middleware')
    process_list['h1'] = hosts['h1'].popen('python ../monax_server.py')   
    process_list['h2'] = hosts['h2'].popen('python ../monax_client.py')
    process_list['h3'] = hosts['h3'].popen('python ../monax_middleware_proxy.py')

    # CLI(net)
    for host, line in pmonitor(process_list):
        if host:
            print( "<%s>: %s" % ( host, line ) )
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    Testing()