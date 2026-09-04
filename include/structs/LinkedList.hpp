#pragma once

#include <iostream>

namespace structs {

template <typename T> class LLNode {
public:
    LLNode(const T& data) : m_data(data) {}

    LLNode* getNext() const { return m_next; }
    LLNode* getPrev() const { return m_prev; }
    LLNode* setNext(LLNode* node) { m_next = node; return node; }
    LLNode* setPrev(LLNode* node) { m_prev = node; return node; }

    T& data() { return m_data; }
    const T& data() const { return m_data; }

    friend std::ostream& operator<<(
        std::ostream& os, 
        const LLNode<T>* node
    ) { 
        if (node)   { os << node->m_data;   return os; }
        else        { os << "nullptr";      return os; }
    }

protected:
    LLNode* m_next = nullptr;
    LLNode* m_prev = nullptr;
    T m_data;
};

template <typename T> struct LinkedList {
public:
    LinkedList() = default;
    ~LinkedList() { cleanup(); }

    LinkedList(const LinkedList& ll) {
        for (auto* n = ll.getRoot(); n; n = n->getNext()) append(n->data());
    }

    LinkedList& operator=(LinkedList ll) {
        std::swap(m_root, ll.m_root);
        std::swap(m_tail, ll.m_tail);
        std::swap(m_size, ll.m_size);
        return *this;
    }

    LLNode<T>* getRoot() const { return m_root; }
    LLNode<T>* getTail() const { return m_tail; }
    size_t size() const { return m_size; }

    LinkedList& append(const T& data) {
        if (m_root == nullptr) { init(data); return *this; }

        auto prevTail = m_tail;
        m_tail->setNext(new LLNode<T>(data));
        m_tail = m_tail->getNext();
        m_tail->setPrev(prevTail);
        ++m_size;

        return *this;
    }

    LinkedList& prepend(const T& data) {
        if (m_root == nullptr) { init(data); return *this; }

        auto prevRoot = m_root;
        m_root->setPrev(new LLNode<T>(data));
        m_root = m_root->getPrev();
        m_root->setNext(prevRoot);
        ++m_size;

        return *this;
    }

    LinkedList& erase(const size_t index) {
        LLNode<T>* node = at(index);

        LLNode<T>* prev = node->getPrev();
        LLNode<T>* next = node->getNext();

        // 4 cases: prev and next, prev and !next, !prev and next, !prev and !next
        if      ( prev &&  next)    { prev->setNext(next); next->setPrev(prev); delete node; }
        else if ( prev && !next)    { m_tail = prev; m_tail->setNext(nullptr);  delete node; }
        else if (!prev &&  next)    { m_root = next; m_root->setPrev(nullptr);  delete node; }
        else                        { m_tail = nullptr; m_root = nullptr;       delete node; }

        --m_size;

        return *this;
    }

    LLNode<T>* at(const size_t i) const {
        if (i > (m_size - 1) || m_size == 0)
            throw std::out_of_range(
                "Index [" + std::to_string(i) + "] is out of range."
            );

        int dir = (i < m_size / 2);
        size_t idx = (!dir) * (m_size - 1);
        LLNode<T>* curr = dir ? m_root : m_tail;
        int inc = (dir * 2) - 1;

        for (size_t start = idx; start != i; start += inc)
            curr = dir ? curr->getNext() : curr->getPrev();

        return curr;
    }

    LLNode<T>* operator[](const size_t i) const { return at(i); }

    class Iterator {
    public:
        explicit Iterator(LLNode<T>* node) : m_node(node) {}
        
        T& operator*() const { return m_node->data(); }
        Iterator& operator++() { m_node = m_node->getNext(); return *this; }
        bool operator==(const Iterator& o) const { return m_node == o.m_node; }

    private:
        LLNode<T>* m_node;
    };

    Iterator begin()    { return Iterator(m_root); }
    Iterator end()      { return Iterator(nullptr); }

private:
    LLNode<T>* m_root = nullptr;
    LLNode<T>* m_tail = nullptr;
    size_t m_size = 0;

    void init(const T& data) {
        m_root = new LLNode<T>(data);
        m_tail = m_root;
        ++m_size;
    }

    void cleanup() {
        LLNode<T>* curr = m_root;
        
        
        while (curr) {
            auto next = curr->getNext();
            delete curr;
            curr = next;
        }
    }
};

}