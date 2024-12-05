USER = $(shell whoami)
UID = $(shell id -u)
GID = $(shell id -g)
current_dir = $(shell pwd)
project-name = primitives

########################################################################################
##################################### RUN COMMANDS #####################################
########################################################################################

run-amiga-listener:
	docker run \
			--rm \
			--name ${project-name}-amiga \
			--privileged \
			--net=host \
			-v ${current_dir}:/amiga \
			-v /dev:/dev \
			-v ${current_dir}/data:/data \
			-v ${current_dir}/resources/.bash_history:/root/.bash_history \
			-it \
			--cap-add SYS_NICE \
			--ulimit rtprio=98 \
			--ulimit memlock=-1 \
			--ulimit rttime=-1 \
			${project-name}-amiga:latest bash -c "pip install -e . && python -m amiga --zmq --cfg cfg/amiga.yaml"


run-zed:
	@docker run \
			--runtime=nvidia \
			--rm \
			--name ${project-name}-zed \
			--privileged \
			--net=host \
			-v ${current_dir}:/amiga \
			-v /dev:/dev \
			-v ${current_dir}/data:/data \
			-v ${current_dir}/resources/.bash_history:/root/.bash_history \
			-v ${current_dir}/resources/zed/zed_calib/:/usr/local/zed/settings/ \
			-v ${current_dir}/resources/zed/zed_resources:/usr/local/zed/resources/ \
			-e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
			-it \
			${project-name}-zed:latest bash -c "pip install -e . && python -m amiga --zmq --cfg cfg/zed.yaml"


run-zed-obj-det:
	@docker run \
		--runtime=nvidia \
		--rm \
		--privileged \
		--name ${project-name}-zed-obj-det \
		--net=host \
		-v ${current_dir}:/amiga \
		--user ${UID}:${GID} \
		-it \
		${project-name}-torch bash -c "pip install -e . && python -m amiga.tools.detect_objects --cfg cfg/zed.yaml --weights resources/models/241128_yolov11s_datav4_mAP0.5=0.815.pt"

run-dev:
	@docker run \
		--runtime=nvidia \
		--rm \
		--privileged \
		--name ${project-name}-dev \
		--net=host \
		-v ${current_dir}:/amiga \
		--user ${UID}:${GID} \
		-it \
		${project-name}-torch bash -c "pip install transforms3d && pip install -e . && python -m amiga --script --cfg cfg/scripts/dev.yaml"

run-realsense:
	@docker run \
		--runtime=nvidia \
		--rm \
		--name ${project-name}-rs \
		--privileged \
		--net=host \
		-v ${current_dir}:/amiga \
		-v /dev:/dev \
		-v ${current_dir}/data:/data \
		-v ${current_dir}/resources/.bash_history:/root/.bash_history \
		-e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
		-it \
		${project-name}-rs:latest bash -c "pip install -e . && python -m amiga --zmq --cfg cfg/realsense.yaml"


run-handeye:
	@docker run \
		--runtime=nvidia \
		--rm \
		--privileged \
		--name ${project-name}-handeye \
		--net=host \
		-v ${current_dir}:/amiga \
		--user ${UID}:${GID} \
		-it \
		${project-name}-user bash -c "pip install -e . && python -m amiga.tools.eye_in_hand --cfg cfg/tools/eyeinhand.yaml

.robot_tool:
	@docker run \
		--runtime=nvidia \
		--rm \
		--privileged \
		--name ${project-name}-gripper \
		--net=host \
		-v ${current_dir}:/amiga \
		--user ${UID}:${GID} \
		-it \
		${project-name}-user bash -c "pip install -e . && python -m amiga.tools.robot --cfg cfg/amiga.yaml ${rob_flag}"

train-grasp:
	@docker run \
		--runtime=nvidia \
		--rm \
		--privileged \
		--name ${project-name}-train-grasp \
		--net=host \
		-v ${current_dir}:/amiga \
		-v ${current_dir}/resources/cache:/home/${USER}/.cache \
		-v ${current_dir}/resources/.bash_history:/home/${USER}/.bash_history \
		-v ${current_dir}/resources/.netrc:/home/${USER}/.netrc \
		--user ${UID}:${GID} \
		-it \
		${project-name}-torch bash -c "pip install -e . && python -m amiga.tools.learning_grasp_params --cfg cfg/tools/grasp_params.yaml ${rob_flag}"


open-gripper: rob_flag=--open
open-gripper: .robot_tool

close-gripper: rob_flag=--close
close-gripper: .robot_tool

freeddrive-on: rob_flag=--freedrive
freeddrive-on: .robot_tool

freeddrive-off: rob_flag=--no-freedrive
freeddrive-off: .robot_tool

home: rob_flag=--home
home: .robot_tool


stop:
	@docker stop ${project-name}-zed 2> /dev/null || true
	@docker stop ${project-name}-amiga 2> /dev/null || true
	@docker stop ${project-name}-rs 2> /dev/null || true

exec-amiga:
	docker exec -it ${project-name}-amiga bash

exec-zed:
	docker exec -it ${project-name}-zed bash

exec-torch:
	docker exec -it ${project-name}-torch bash


get-zed-img:
	docker run \
		--runtime=nvidia \
		--rm \
		--name ${project-name}-tools \
		--privileged \
		--net=host \
		-v ${current_dir}:/amiga \
		--user ${UID}:${GID} \
		-it \
		${project-name}-user bash -c "pip install -e . && python -m amiga.tools.get_img --cfg cfg/zed.yaml"


get-rs-img:
	docker run \
		--runtime=nvidia \
		--rm \
		--name ${project-name}-tools \
		--privileged \
		--net=host \
		-v ${current_dir}:/amiga \
		--user ${UID}:${GID} \
		-it \
		${project-name}-user bash -c "pip install -e . && python -m amiga.tools.get_img --cfg cfg/realsense.yaml"


########################################################################################
#################################### BUILD COMMANDS ####################################
########################################################################################
.build-base:
	@touch ${current_dir}/resources/.bash_history
	@touch ${current_dir}/resources/.netrc
	@mkdir -p ${current_dir}/resources/cache/
	docker build -f dockerfiles/Dockerfile.base -t ${project-name}-base:latest .

build-zed: .build-base
	@docker build -f dockerfiles/Dockerfile.zed -t ${project-name}-zed:latest .

build-realsense: .build-base
	@docker build -f dockerfiles/Dockerfile.realsense -t ${project-name}-rs:latest .

build-amiga-listener: .build-base
	@docker build -f dockerfiles/Dockerfile.amiga -t ${project-name}-amiga:latest .

build-torch: .build-base
	@docker build -f dockerfiles/Dockerfile.torch -t ${project-name}-torch:latest --build-arg USERNAME=${USER} --build-arg UID=${UID} --build-arg GID=${GID} .

build-localuser: .build-base
	@docker build -f dockerfiles/Dockerfile.localuser -t ${project-name}-user:latest --build-arg USERNAME=${USER} --build-arg UID=${UID} --build-arg GID=${GID} .
