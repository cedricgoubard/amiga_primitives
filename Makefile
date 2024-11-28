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
			--name primitives-amiga \
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
			primitives-amiga:latest bash -c "pip install --root-user-action -e . && python -m amiga --zmq --cfg cfg/amiga.yaml"


run-zed:
	@docker run \
			--runtime=nvidia \
			--rm \
			--name primitives-zed \
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
			primitives-zed:latest bash -c "pip install -e . && python -m amiga --zmq --cfg cfg/zed.yaml"


run-realsense:
	@docker run \
			--runtime=nvidia \
			--rm \
			--name primitives-rs \
			--privileged \
			--net=host \
			-v ${current_dir}:/amiga \
			-v /dev:/dev \
			-v ${current_dir}/data:/data \
			-v ${current_dir}/resources/.bash_history:/root/.bash_history \
			-e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
			-it \
			primitives-rs:latest bash -c "pip install -e . && python -m amiga --zmq --cfg cfg/realsense.yaml"

stop:
	@docker stop primitives-zed 2> /dev/null || true
	@docker stop primitives-amiga 2> /dev/null || true
	@docker stop primitives-rs 2> /dev/null || true

exec-amiga:
	docker exec -it primitives-amiga bash

exec-zed:
	docker exec -it primitives-zed bash

exec-torch:
	docker exec -it primitives-torch bash


get-zed-img:
	docker run \
		--runtime=nvidia \
		--rm \
		--name primitives-tools \
		--privileged \
		--net=host \
		-v ${current_dir}:/amiga \
		--user ${UID}:${GID} \
		-it \
		primitives-user bash -c "pip install -e . && python -m amiga.tools.get_img --cfg cfg/zed.yaml"


get-rs-img:
	docker run \
		--runtime=nvidia \
		--rm \
		--name primitives-tools \
		--privileged \
		--net=host \
		-v ${current_dir}:/amiga \
		--user ${UID}:${GID} \
		-it \
		primitives-user bash -c "pip install -e . && python -m amiga.tools.get_img --cfg cfg/realsense.yaml"


########################################################################################
#################################### BUILD COMMANDS ####################################
########################################################################################
.build-base:
	@touch ${current_dir}/resources/.bash_history
	@touch ${current_dir}/resources/.netrc
	docker build -f dockerfiles/Dockerfile.base -t primitives-base:latest .

build-zed: .build-base
	@docker build -f dockerfiles/Dockerfile.zed -t primitives-zed:latest .

build-realsense: .build-base
	@docker build -f dockerfiles/Dockerfile.realsense -t primitives-rs:latest .

build-amiga-listener: .build-base
	@docker build -f dockerfiles/Dockerfile.amiga -t primitives-amiga:latest .

build-torch: .build-base
	@docker build -f dockerfiles/Dockerfile.torch -t primitives-torch:latest --build-arg USERNAME=${USER} --build-arg UID=${UID} --build-arg GID=${GID} .

build-localuser: .build-base
	@docker build -f dockerfiles/Dockerfile.localuser -t primitives-user:latest --build-arg USERNAME=${USER} --build-arg UID=${UID} --build-arg GID=${GID} .
