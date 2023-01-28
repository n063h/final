scp -P $port -r  root@region-11.autodl.com:/root/autodl-tmp/datasets /root/autodl-tmp
mkdir /root/autodl-tmp/checkpoints
mkdir /root/tf-logs

cat << EOF >> /etc/hosts
151.101.1.194 github.global.ssl.fastly.net
140.82.114.3 github.com
EOF

echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCdYWT7eLypD9P4LfT2a7JgVqYxKJ1jlZb0CIq8/Av7Zq3cIdRHpu4T57FQMyAruLMzYnC/IAR3ml7CMy2kc0AEKb7+DpNGs7O+nLZFjlfubGt1Vdk/4tOJKYNa9NU7EC12mZxzHc6tzBlFM7pzK70XiKebE425RGb6TlixtDVZc7xiWMP/WTQzMEwNOvyAbcqlRSDe3kaWFl+awkd6NsS6c9/xxu1NK5ROcK8L10VpI8DqsWPhB+S0JUiqRCEhGegK7jQXNDMMtemkD81eFsMWW8WGUZYyxXRwB/x40x+t56YuCtBw5njP5UzScnU0yuKAq2ynoky0YZ5nqvmnm5JD0ORQjNNv1x5Iw8kdqXR3hIQz8eqXDP90csQGH+BFqSen+OAu61ZwapbTjoooCnMhPChYyVPpiTQBl887yg2+xckH9L5PF/IMoWuXarWS6K1BQpbxaMX7M96hSNagy5YxBxnIW5y3KjpTKn8Wg19aAXnwsm/FZ7ebZ2lq9AROG2M= 89439@HangsPC" >> ~/.ssh/authorized_keys