export const ODOM_HISTORY = 100;

export enum Sensor {
    CAMERA = "sensors/camera",
    IMU = "sensors/odometry",
}

export interface OdometryItem {
    acc: Float32Array;
    gyro: Float32Array;
    angle: Float32Array;
}
