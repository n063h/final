scp -P $port  root@region-11.autodl.com:/etc/hosts /etc
scp -P $port -r  root@region-11.autodl.com:/root/autodl-tmp/datasets /root/autodl-tmp
mkdir /root/autodl-tmp/checkpoints
mkdir /root/tf-logs