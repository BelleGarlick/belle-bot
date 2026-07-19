# Belle Bot

Belle bot is comprised of numerous microservices. Each will work in tandem to create a little pet robot. 
This way each micro-service is responsible solely for it's one thing.

## Core Components

### Fabric
The **Fabric** is the central communications backbone of Belle Bot. It uses a publish-subscribe model (via WebSockets and REST) to allow different microservices to communicate.

**The Fabric must be running for other modules to function correctly**, as it facilitates all inter-service data exchange. 

For more details on how to run and use the Fabric, see the [Fabric README](belle_bot/infra/fabric/readme.md).

### Sensors
Sensors provide the robot with data from the environment.

- **Camera**: Captures and publishes video frames. See [Camera README](belle_bot/sensors/cameras/readme.md).
- **Microphone**: Captures and publishes audio data. See [Microphone README](belle_bot/sensors/microphone/readme.md).

## TODO
- Create a script to trigger all the scripts including the fabric to run everything
- Should have a way to have the logging run change randomly so that we can upload 30min chunks to blob store if on the same network
- Should have a way to remotely deploy to it
- Have a way to override the host url so that it can stream information remotely and have the compute on another device on the network
- Log system usage metrics and performance logs for each system
- Create pose estimation for face detection and eventually recognition

## Future Components
- Depth handling in the existing camera module which publishes the depth map everywhere
- Local Mapper
  - This should be used to work out the local environment that it can currently see. So it should show information that it can see at the current time step both for humans and for terrain
- Global Mapper
  - This should take information from the local mapper and persist data through time this way we can understand where information is at a global level, not just local
- The LLM module 
  - this should have text output to understand the world around it with basic actions to speak, move cameras, set waypoint to move, perform specific actions like spinning around, save information about a user, save information to local temporary memory, save information to a specific location so when they're near it, tells you information about it
- Movement pipeline
- Have a personal knowledge pipeline
- Have a personality pipeline
- have a human knowledge pipeline
- have a global knowledge pipeline