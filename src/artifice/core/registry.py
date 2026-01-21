"""
Node registration system.

Provides a global registry for discovering and instantiating nodes,
plus a decorator for easy node registration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Type

if TYPE_CHECKING:
    from artifice.core.node import Node


class NodeRegistry:
    """
    Global registry of available node types.

    Nodes are registered by their class name and can be looked up
    for instantiation or serialization purposes.

    Usage:
        # Register a node class
        NodeRegistry.register(MyNode)

        # Or use the decorator
        @register_node
        class MyNode(Node):
            ...

        # Look up a node class
        node_class = NodeRegistry.get("MyNode")

        # Create an instance
        node = NodeRegistry.create("MyNode")
    """

    _registry: dict[str, Type[Node]] = {}
    _categories: dict[str, list[str]] = {}

    @classmethod
    def register(cls, node_class: Type[Node]) -> Type[Node]:
        """
        Register a node class.

        Args:
            node_class: The node class to register

        Returns:
            The registered class (for decorator use)
        """
        name = node_class.__name__

        if name in cls._registry:
            # Allow re-registration (useful during development)
            pass

        cls._registry[name] = node_class

        # Track by category
        category = getattr(node_class, "category", "Utility")
        if category not in cls._categories:
            cls._categories[category] = []
        if name not in cls._categories[category]:
            cls._categories[category].append(name)

        return node_class

    @classmethod
    def unregister(cls, node_class_or_name: Type[Node] | str) -> bool:
        """
        Unregister a node class.

        Args:
            node_class_or_name: Class or class name to unregister

        Returns:
            True if unregistered, False if not found
        """
        name = (
            node_class_or_name
            if isinstance(node_class_or_name, str)
            else node_class_or_name.__name__
        )

        if name not in cls._registry:
            return False

        node_class = cls._registry[name]
        category = getattr(node_class, "category", "Utility")

        del cls._registry[name]

        if category in cls._categories and name in cls._categories[category]:
            cls._categories[category].remove(name)

        return True

    @classmethod
    def get(cls, name: str) -> Type[Node] | None:
        """
        Get a node class by name.

        Args:
            name: Class name of the node

        Returns:
            Node class or None if not found
        """
        return cls._registry.get(name)

    @classmethod
    def create(cls, name: str) -> Node | None:
        """
        Create a node instance by class name.

        Args:
            name: Class name of the node

        Returns:
            New node instance or None if class not found
        """
        node_class = cls._registry.get(name)
        if node_class is not None:
            return node_class()
        return None

    @classmethod
    def get_registry(cls) -> dict[str, Type[Node]]:
        """
        Get the full registry dictionary.

        Returns:
            Dictionary mapping names to node classes
        """
        return cls._registry.copy()

    @classmethod
    def get_categories(cls) -> dict[str, list[str]]:
        """
        Get nodes organized by category.

        Returns:
            Dictionary mapping category names to lists of node names
        """
        return {cat: list(nodes) for cat, nodes in cls._categories.items()}

    @classmethod
    def get_by_category(cls, category: str) -> list[Type[Node]]:
        """
        Get all node classes in a category.

        Args:
            category: Category name

        Returns:
            List of node classes
        """
        names = cls._categories.get(category, [])
        return [cls._registry[name] for name in names if name in cls._registry]

    @classmethod
    def list_all(cls) -> list[str]:
        """
        List all registered node names.

        Returns:
            List of node class names
        """
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (useful for testing)."""
        cls._registry.clear()
        cls._categories.clear()

    @classmethod
    def get_node_info(cls, name: str) -> dict | None:
        """
        Get metadata about a registered node.

        Args:
            name: Node class name

        Returns:
            Dictionary with node info or None if not found
        """
        node_class = cls._registry.get(name)
        if node_class is None:
            return None

        return {
            "name": getattr(node_class, "name", name),
            "class_name": name,
            "category": getattr(node_class, "category", "Utility"),
            "description": getattr(node_class, "description", ""),
            "icon": getattr(node_class, "icon", None),
        }


def register_node(cls: Type[Node]) -> Type[Node]:
    """
    Decorator to register a node class.

    Usage:
        @register_node
        class MyNode(Node):
            name = "My Node"
            category = "Effects"
            ...
    """
    return NodeRegistry.register(cls)


def get_registry() -> NodeRegistry:
    """
    Get the global node registry.

    Returns:
        The NodeRegistry class (which has class methods for all operations)
    """
    return NodeRegistry


def register_nodes_from_module(module) -> int:
    """
    Register all Node subclasses from a module.

    Args:
        module: Python module to scan

    Returns:
        Number of nodes registered
    """
    from artifice.core.node import Node

    count = 0
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, Node)
            and obj is not Node
            and not getattr(obj, "_abstract", False)
        ):
            NodeRegistry.register(obj)
            count += 1

    return count
