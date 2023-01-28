scp -P $port -r  root@region-11.autodl.com:/root/autodl-tmp/datasets /root/autodl-tmp
mkdir /root/autodl-tmp/checkpoints
mkdir /root/tf-logs

cat << EOF >> /etc/hosts
151.101.1.194 github.global.ssl.fastly.net
140.82.114.3 github.com
EOF