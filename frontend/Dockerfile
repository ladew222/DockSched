# Step 1: Build stage
# Use an official Node.js runtime as a parent image
FROM node:alpine as build

# Set the working directory in the Docker container
WORKDIR /app

# Copy the package.json and package-lock.json (or yarn.lock)
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of your app's source code
COPY . .

# Build your app
RUN npm run build

# Step 2: Serve stage
# Use an official nginx image to serve the app
FROM nginx:alpine

# Copy the build output to replace the default nginx contents.
COPY --from=build /app/build /usr/share/nginx/html

# Expose port 80 to the Docker host, so we can access it
# from the outside.
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
