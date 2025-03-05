
#include <opencv4/opencv2/opencv.hpp>

class CV_EXPORTS _TickMeter {
public:
	_TickMeter();
	void start();
	void stop();

	int64 getTimeTicks() const;
	double getTimeMicro() const;
	double getTimeMilli() const;
	double getTimeSec()   const;
	int64 getCounter() const;

	void reset();
private:
	int64 counter;
	int64 sumTime;
	int64 startTime;
};

std::ostream& operator << (std::ostream& out, const _TickMeter& tm);


_TickMeter::_TickMeter() { reset(); }
int64 _TickMeter::getTimeTicks() const { return sumTime; }
double _TickMeter::getTimeMicro() const { return  getTimeMilli() * 1e3; }
double _TickMeter::getTimeMilli() const { return getTimeSec() * 1e3; }
double _TickMeter::getTimeSec() const { return (double)getTimeTicks() / cv::getTickFrequency(); }
int64 _TickMeter::getCounter() const { return counter; }
void _TickMeter::reset() { startTime = 0; sumTime = 0; counter = 0; }

void _TickMeter::start() { startTime = cv::getTickCount(); }
void _TickMeter::stop()
{
	int64 time = cv::getTickCount();
	if (startTime == 0)
		return;
	++counter;
	sumTime += (time - startTime);
	startTime = 0;
}

std::ostream& operator << (std::ostream& out, const _TickMeter& tm) { return out << tm.getTimeSec() << "sec"; }