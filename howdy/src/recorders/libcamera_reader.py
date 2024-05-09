from cv2 import cvtColor, \
		COLOR_GRAY2BGR, \
		COLOR_YUV2BGR_YUYV, \
		CAP_PROP_FRAME_WIDTH, \
		CAP_PROP_FRAME_HEIGHT, \
		CAP_PROP_FRAME_WIDTH, \
		CAP_PROP_FRAME_HEIGHT
import libcamera as lc
from MappedFrameBuffer import MappedFrameBuffer
import numpy as np
import selectors
from typing import Tuple

# A map of libcamera.PixelFormats to cv2 ColorConversionCodes
pixelformat_2_csc = {
	lc.PixelFormat('NV12').fourcc: COLOR_YUV2BGR_NV12
}

class libcamera_reader:
	def __init__(self, camera_id):
		self.camera_id = camera_id
		self.height = 0
		self.width = 0
		self.mfbs = {}
		self.probe()

	def set(self, prop, setting):
		""" Setter method for height and width """
		if prop == CAP_PROP_FRAME_WIDTH:
			self.width = setting
		elif prop == CAP_PROP_FRAME_HEIGHT:
			self.height = setting

	def get(self, prop):
		""" Getter method for height and width """
		if prop == CAP_PROP_FRAME_WIDTH:
			return self.width
		elif prop == CAP_PROP_FRAME_HEIGHT:
			return self.height

	def probe(self):
		self.cm = lc.CameraManager.singleton()

		if not self.cm.cameras:
			raise Exception("No cameras identified on the system")

		self.camera = self.cm.get(self.camera_id)

		if not self.camera:
			raise Exception(f"Camera ID {self.camera_id} not found")

		self.camera.acquire()

		# libcamera cameras can provide a default configuration for a
		# number of stream roles. We generate such a default and use the
		# provided width and height to set properties.

		config = self.camera.generate_configuration([
				lc.StreamRole.Viewfinder
			])

		stream_config = config.at(0)

		self.set(CAP_PROP_FRAME_WIDTH, int(stream_config.size.width))
		self.set(CAP_PROP_FRAME_HEIGHT, int(stream_config.size.height))

		# We could have any pixel format by default, we need to use one
		# that we know can be handled by cv2. Roll through the cameras
		# formats and pick the first one that has an entry in the
		# pixelformat_2_csc map of pixel formats to color conversion
		# codes.
		pixel_format = None
		for fmt in stream_config.formats.pixel_formats:
			if fmt.fourcc in pixelformat_2_csc.keys():
				pixel_format = fmt
				break

		assert(pixel_format)

		# libcamera requires that altered stream configurations be
		# validated before configuring the camera with them. The
		# validation will _always_ return a working config but howdy may
		# not support whatever format was reverted to, so if the status
		# is anything other than 0 just fail here.
		#
		# \todo can this be handled a bit better?
		stream_config.pixel_format = pixel_format
		ret = config.validate()
		assert(ret.value == 0)

		self.camera.configure(config)

		self.stream = stream_config.stream


	def record(self):
		"""
		Start recording. This involves allocating buffers through the
		libcamera FrameBufferAllocator class and creating a bunch of
		Requests, starting the camera and queuing the Requests to it. At
		the conclusion of this function the camera should be recording
		and filling Requests with data ready to be retrieved in read()
		"""

		allocator = lc.FrameBufferAllocator(self.camera)

		ret = allocator.allocate(self.stream)
		assert ret > 0

		num_bufs = len(allocator.buffers(self.stream))

		reqs = []
		for i in range(num_bufs):
			req = self.camera.create_request(i)

			buffer = allocator.buffers(self.stream)[i]
			req.add_buffer(self.stream, buffer)

			reqs.append(req)

			# We need to mmap() the buffer so that we can read the
			# image data from it later on.
			self.mfbs[buffer] = MappedFrameBuffer(buffer).mmap()

		self.camera.start()

		for req in reqs:
			self.camera.queue_request(req)

		self.started = True


	def grab(self):
		""" Read a single frame from the camera. """
		self.read()

	def _get_frame_data(self, req: lc.Request):
		"""
		Get the frame data from a libcamera request
		"""

		buffers = req.buffers

		# We only configure a single stream so we should only have a
		# single buffer. Make sure that's true and retrieve it.
		assert len(buffers) == 1
		_, fb = next(iter(buffers.items()))

		mfb = self.mfbs[fb]

		# We are only supporting formats with a single plane here, which
		# ought to be properly validated.
		#
		# \todo Handle planar formats
		return mfb.planes[0]

	def read(self):
		"""
		Read a single frame from the camera. The calling function is
		VideoCapture.read_frame(), which expects to receive data that it
		can convert using cvtColor(frame, cv2.COLOR_BGR2GRAY), which
		means that we need to check the format in use and convert it to
		BGR.
		"""

		if not self.started:
			self.record()

		# cm.get_ready_requests() does not block, so we use a Selector
		# to wait for a camera event.

		sel = selectors.DefaultSelector()
		sel.register(self.cm.event_fd, selectors.EVENT_READ)

		# \todo 5s timeout is preposterously long; reconsider if this is
		# actually necessary
		while not sel.select(5):
			continue

		reqs = self.cm.get_ready_requests()

		# There might be multiple Requests ready at this point...but we
		# just want the data from the first. We do want to return the
		# other Requests to the camera though so that they can be filled
		# again with new data, so set a flag and just recycle anything
		# if it's true.

		frame_data = None

		for req in reqs:

			if frame_data is None:
				frame_data = self._get_frame_data(req)

			# Now that we have what we need, we need to requeue the
			# request so that it can be filled again.
			req.reuse()
			self.camera.queue_request(req)

		# Convert the raw frame_date to a numpy array
		img = (np.frombuffer(frame_data, np.uint8))

		# Hack Warning!
		# Currently this is being used with Surface devices and the IPU3
		# which produces data in NV12...because I haven't gotten the
		# cv2 conversions working properly from NV12 natively, I'm just
		# slicing out the Y plane and treating it as a gray image. That
		# is wrong and hacky...but it works. The Y plane is a single
		# byte per pixel so easy to figure out.
		img = img[:int(self.width * self.height)]

		# Convert the numpy array to a BGR image with cv2. To know the
		# proper conversion code to use we need to know the pixel format
		# of the stream. We should use that as a key for the map of
		# pixel formats to conversion codes above, but since I haven't
		# gotten that working yet and we're slicing out a gray image we
		# can just hardcode that code.
		pixel_format = self.stream.configuration.pixel_format
		img_bgr = cvtColor(img, COLOR_GRAY2BGR)

		# Convert the grayscale image array into a proper RGB style numpy array
		img2 = (np.frombuffer(img_bgr, np.uint8)\
			  .reshape([self.width, self.height, 3]))

		# Return a single frame of video
		return 0, img2

	def release(self):
		""" Empty our array. If we had a hold on the camera, we would give it back here. """
		self.camera.stop()
		self.camera.release()

		self.mfbs = {}
