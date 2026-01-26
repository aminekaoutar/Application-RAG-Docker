@startuml
!theme sunlust
title Architecture DevOps du Projet

node "Docker Host" {
    component "Cloudflared" as Tunnel #Orange
    component "Caddy" as Proxy #Yellow
    
    package "Internal Network" {
        component "Web App" as Web
        component "Database" as DB
        component "Portainer" as Admin
    }
}

cloud "Internet" {
    actor User
}

User --> Tunnel : HTTPS
Tunnel --> Proxy : HTTP
Proxy --> Web : /
Proxy --> Admin : /admin
Web --> DB
@enduml