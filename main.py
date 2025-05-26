from beamngpy import BeamNGpy, Scenario, Vehicle

bng = BeamNGpy(
    'localhost',
    25252,
    home=r'C:\Users\Sivar\Desktop\BeamNG.tech.v0.35.5.0',
    user=r'C:\Users\Sivar\Desktop\BeamNG.tech.v0.35.5.0\userFolder'
)

bng.open()

scenario = Scenario('west_coast_usa', 'example')
vehicle = Vehicle('ego_vehicle', model='etk800', license='PYTHON')

scenario.add_vehicle(vehicle, pos=(-717, 101, 118), rot_quat=(0, 0, 0.3826834, 0.9238795))
scenario.make(bng)

bng.scenario.load(scenario)
bng.scenario.start()

vehicle.ai.set_mode('traffic')
input('Hit Enter when done...')

bng.disconnect()
