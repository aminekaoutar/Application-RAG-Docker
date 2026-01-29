FROM caddy:2.10

COPY Caddyfile /etc/caddy/Caddyfile

EXPOSE 8081 2020

CMD ["caddy", "run", "--config", "/etc/caddy/Caddyfile"]