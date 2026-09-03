#pragma once

namespace structs {

template <typename T> class LLNode {
public:
    LLNode(const T& data) : m_data(data) {}
    ~LLNode() { if (m_next != nullptr) delete m_next; }

    LLNode* getNext() const { return m_next; }
    LLNode* getPrev() const { return m_prev; }
    void setNext(LLNode* node) { m_next = node; }
    void setPrev(LLNode* node) { m_prev = node; }

    T data() const { return m_data; }

protected:
    LLNode* m_next = nullptr;
    LLNode* m_prev = nullptr;
    T m_data;
};

template <typename T> struct LinkedList {
public:
    LinkedList() = default;
    // LinkedList(const T& data) : m_root(new LLNode<T>(data)) { m_tail = m_root; }
    ~LinkedList() { if (m_root != nullptr) delete m_root; }

    LLNode<T>* getRoot() const { return m_root; }
    LLNode<T>* getTail() const { return m_tail; }
    size_t size() const { return m_size; }

    void append(const T& data) {
        if (m_root == nullptr) {
            m_root = new LLNode<T>(data);
            m_tail = m_root;
            m_root->setNext(m_tail);
            m_tail->setPrev(m_root);
            ++m_size;
            return;
        }

        m_tail->setNext(new LLNode<T>(data));
        m_tail->getNext()->setPrev(m_tail);
        m_tail = m_tail->getNext();
        ++m_size;
    }

    void prepend(const T& data) {
        if (m_root == nullptr) {
            m_root->setNext(new LLNode<T>(data));
            m_tail = m_root;
            m_root->setNext(m_tail);
            m_tail->setPrev(m_root);
            ++m_size;
            return;
        }

        m_root->setPrev(new LLNode<T>(data));
        m_root->getPrev()->setNext(m_root);
        m_root = m_root->getPrev();
        ++m_size;
    }

    T operator[](const int i) {
        LLNode<T>* curr = nullptr;
        bool forward = true;
        int idx;

        if (m_root == nullptr) return nullptr;
        else curr = m_root;

        if (i > (m_size - 1) / 2) {
            forward = false;
            idx = m_size - 1;
        } else idx = 0;

        if (forward) {
            while(curr != nullptr) {
                if (idx == i) return curr.data();
                curr = curr->getNext();
                ++idx;
            }
        }
        else {
            while(curr != nullptr) {
                if (idx == i) return curr.data();
                curr = curr->getPrev();
                --idx;
            }
        }
    }

private:
    LLNode<T>* m_root = nullptr;
    LLNode<T>* m_tail = nullptr;
    size_t m_size = 0;
};

}