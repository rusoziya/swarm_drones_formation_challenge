# generated from gazebo_plugins/env-hooks/gazebo_plugins.sh.in

# detect if running on Darwin platform
_UNAME=`uname -s`
_IS_DARWIN=0
if [ "$_UNAME" = "Darwin" ]; then
  _IS_DARWIN=1
fi
unset _UNAME

if [ $_IS_DARWIN -eq 0 ]; then
  ament_prepend_unique_value LD_LIBRARY_PATH 
else
  ament_prepend_unique_value DYLD_LIBRARY_PATH 
fi
unset _IS_DARWIN

ament_prepend_unique_value GAZEBO_PLUGIN_PATH "$AMENT_CURRENT_PREFIX/lib"

