FROM cedricmelancon/ros2_ai:latest

COPY requirements.txt /main/requirements.txt
COPY run_training.sh /main/run_training.sh
RUN chmod +x /main/run_training.sh

WORKDIR /workspace/src/info_odometry
ENTRYPOINT ["/main/run_training.sh"]
CMD ["-- help"]