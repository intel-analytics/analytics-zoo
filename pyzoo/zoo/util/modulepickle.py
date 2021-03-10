import os
from io import BytesIO
from tarfile import TarFile
import tempfile
import importlib
import importlib.machinery
import hashlib
import sys
from logging import getLogger

__all__ = ('extend', 'extend_ray', 'extend_cloudpickle')

log = getLogger(__name__)

TEMPDIR_ID = 'MODULEPICKLE'


def md5(compressed):
    md5 = hashlib.md5()
    md5.update(compressed)
    return md5.hexdigest()[:16]  # 32 bytes ought to be enough for everyone


class Package:
    def __init__(self, name, compressed):
        self.name = name
        self.compressed = compressed
        self.md5 = md5(compressed)

    def invalidate_caches(self):
        # Chuck out any modules that come from one of our temp dirs, so that when they get importer next time it's imported from
        # the shiny new temp dir
        modules = list(sys.modules)
        for k in modules:
            v = sys.modules[k]
            filepath = getattr(v, '__file__', '') or ''
            if f'{TEMPDIR_ID}-{self.name}-' in filepath:
                del sys.modules[k]

        # And then invalidate the cache of everyone on the meta_path, just to be safe.
        importlib.invalidate_caches()

    def uninstall(self):
        sys.path = [p for p in sys.path if f'{TEMPDIR_ID}-{self.name}-' not in p]

    def extract(self):
        # Salt the temp directory with the hashcode of the compressed dir, so that when the next copy of it comes down the line,
        #  we can either reuse the existing dir if it's the same, or point ourselves at a new one if it isn't.
        dirpath = tempfile.mkdtemp(prefix=f'{TEMPDIR_ID}-{self.name}-{self.md5}-')
        bs = BytesIO(self.compressed)
        with TarFile(fileobj=bs) as tf:
            tf.extractall(os.path.join(dirpath))
        return dirpath

    def install(self):
        """'Installing' this package means extracting it to a hash-salted temp dir and then appending the dir to the path"""
        # Only need to install it if the hash of the dir has changed since we last added it to the path
        if not any(self.md5 in p for p in sys.path):
            self.uninstall()
            self.invalidate_caches()
            sys.path.append(self.extract())

    def load(self, name):
        self.install()
        return importlib.import_module(name)


def compress(packagename, path):
    tar = BytesIO()
    with TarFile(fileobj=tar, mode='w') as tf:
        tf.add(path, packagename)
    # TODO: This was originally gzipped, but the gzipped value seems to change on repeated compressions, breaking hashing.
    # Looks like the issue is a timestamp that can be overriden with a parameter, but let's leave it uncompressed for now.
    return tar.getvalue()


def import_compressed(name, package, class_name):
    res = package.load(name)
    if getattr(res, class_name, None):
        class_type = getattr(res, class_name)
        return class_type.__new__(class_type)
    else:
        return res


def packagename(module):
    # The package we want to zip up is the first part of the module name
    # TODO: Check this holds on relative imports
    return module.__name__.split('.')[0]


def is_local(module):
    # If the module is in the current working directory,
    # and it doesn't have `site-packages` in it's path (which probably means it's part of a local virtualenv)
    # assume it's local and that it's cool to pickle it.
    path = getattr(module, '__file__', None)

    # sys.executable = $python_lib_path/bin/python
    python_lib_path = sys.executable[:-len("/bin/python")]

    if path is None:
        return False

    if path.startswith(python_lib_path):
        return False

    return True


def get_path(module):
    path = getattr(module, '__file__', '')
    package_name = module.__name__
    # Single file
    if "." not in package_name and path.endswith(package_name + ".py"):
        return path
    package_path_part = package_name.replace(".", "/")
    package_first_part = package_path_part.split("/")[0]
    idx = path.index(package_path_part)
    import os
    package_path = os.path.join(path[:idx], package_first_part)
    return package_path


def extend(base):
    """Create a Pickler that can pickle packages by inheriting from `base`

    We're dynamically inheriting from `base` here because my principal use case is extending ray's pickler, and ray's
    installation dependencies are vast. Don't want to truck that around for a one-module package which works just as
    well with cloudpickle.
    """

    class ModulePickler(base):

        dispatch = base.dispatch.copy()

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.packages = {}

        def compress_package(self, name, path):
            # The same package might contain many of the modules a function references, so it makes sense to cache them
            # as we go.
            packagename = name
            if name not in self.packages:
                if path.endswith(".py"):
                    packagename = name + ".py"
                compressed = compress(packagename, path)
                self.packages[name] = Package(name, compressed)
            return self.packages[name]

        def reducer_override(self, obj):
            if getattr(obj, "__module__", None):
                module = sys.modules[obj.__module__]
                if module.__name__ != "__main__" and is_local(module):
                    if packagename(module) in self.packages:
                        package = self.packages[packagename(module)]
                    else:
                        print("get local {} in save_module, path is {}".format(module.__name__, module.__file__))
                        package = self.compress_package(packagename(module), get_path(module))
                    args = (module.__name__, package, obj.__class__.__name__)
                    return import_compressed, args, obj.__dict__
            return super().reducer_override(obj)

    return ModulePickler


def extend_ray():
    """Extends Ray's CloudPickler with a ModulePickler"""
    import ray.cloudpickle
    ray.cloudpickle.CloudPickler = extend(ray.cloudpickle.CloudPickler)
    ray.cloudpickle.dump.__globals__['CloudPickler'] = ray.cloudpickle.CloudPickler
    ray.cloudpickle.dumps.__globals__['CloudPickler'] = ray.cloudpickle.CloudPickler


def extend_cloudpickle():
    """Extends cloudpickle's CloudPickler with a ModulePickler"""
    import cloudpickle
    cloudpickle.CloudPickler = extend(cloudpickle.CloudPickler)
    cloudpickle.dump.__globals__['CloudPickler'] = cloudpickle.CloudPickler
    cloudpickle.dumps.__globals__['CloudPickler'] = cloudpickle.CloudPickler
