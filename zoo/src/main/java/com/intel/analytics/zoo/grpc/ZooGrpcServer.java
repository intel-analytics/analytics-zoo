package com.intel.analytics.zoo.grpc;


import io.grpc.BindableService;
import io.grpc.Server;
import io.grpc.ServerBuilder;

import java.io.IOException;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;


/**
 * Zoo gRPC server class
 * After protobuf generated and service is implemented, service could be passed to ZooGrpcServer
 * to start serving request.
 */
public class ZooGrpcServer {
    private static final Logger logger = Logger.getLogger(ZooGrpcServer.class.getName());
    private final int port;
    private final Server server;
    public ZooGrpcServer(BindableService service) {
        this(8980, "zoo-grpc-conf.yaml", service);
    }
    public ZooGrpcServer(String configPath, BindableService service) {
        this(8980, configPath, service);
    }
    /** Entrypoint of ZooGrpcServer */
    public ZooGrpcServer(int port, String configPath, BindableService service) {
        this.port = port;
        server = ServerBuilder.forPort(port)
                .addService(service)
                .build();
    }

    /** Start serving requests. */
    public void start() throws IOException {
        /* The port on which the server should run */
        server.start();
        logger.info("Server started, listening on " + port);
        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                // Use stderr here since the logger may have been reset by its JVM shutdown hook.
                System.err.println("*** shutting down gRPC server since JVM is shutting down");
                try {
                    ZooGrpcServer.this.stop();
                } catch (InterruptedException e) {
                    e.printStackTrace(System.err);
                }
                System.err.println("*** server shut down");
            }
        });
    }
    public void stop() throws InterruptedException {
        if (server != null) {
            server.shutdown().awaitTermination(30, TimeUnit.SECONDS);
        }
    }
    /**
     * Await termination on the main thread since the grpc library uses daemon threads.
     */
    public void blockUntilShutdown() throws InterruptedException {
        if (server != null) {
            server.awaitTermination();
        }
    }
}
